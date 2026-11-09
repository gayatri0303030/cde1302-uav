"""Explicit Gauss–Newton; zero-damping LSMR and robust-cost backtracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import eigsh, lsmr, svds


@dataclass
class Result:
    state: object
    status: str
    history: list = field(default_factory=list)
    conditioning: dict = field(default_factory=dict)
    covariance: object = None
    solver: str = "explicit_gauss_newton_lsmr_damp_0"


def rank_diagnostics(j):
    if j.shape[0] < j.shape[1]:
        return dict(full_rank=False, reason="fewer residuals than variables")
    # Judge rank on the column-scaled Jacobian (the same scaling the iterations use).  Unscaled,
    # one very tight prior (largest singular value ~1e10, 3 Nov D4/D5) made a perfectly
    # observable graph look singular, and the solver then returned the starting values untouched.
    column_norm = np.sqrt(np.asarray(j.power(2).sum(axis=0)).ravel())
    if np.any(column_norm <= 1e-15):
        return dict(full_rank=False, reason="a variable no factor touches")
    j = j.multiply(1.0 / column_norm).tocsr()
    # ARPACK's smallest-singular-value iteration can stall on well-constrained
    # pose chains. A dense SVD is practical for these moderate offline graphs
    # and diagnoses actual rank rather than iterative-eigensolver convergence.
    if j.shape[1] <= 2000 and j.shape[0] * j.shape[1] <= 16_000_000:
        singular = np.linalg.svd(j.toarray(), compute_uv=False)
        smallest, largest = singular[-1], singular[0]
    else:
        # Larger graphs (longer flights): dense SVD gets slow (~25 s at 3000 variables) and
        # ARPACK's "SM" iteration can fail to converge on a perfectly well-posed graph.
        # Shift-invert on the normal matrix instead: factorising J^T J fails outright if it is
        # singular, otherwise the smallest eigenvalue converges in a few iterations.  Squaring
        # limits what can be resolved to sigma_min/sigma_max ~ 1e-8, so this path uses a 1e-7
        # threshold (condition number < 1e7) instead of 1e-10.
        try:
            a = (j.T @ j).tocsc()
            lam_max = eigsh(a, k=1, which="LA", return_eigenvectors=False)[0]
            lam_min = eigsh(a, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0]
        except Exception as exc:
            return dict(full_rank=False, reason=f"rank verification failed (normal matrix singular?): {exc}")
        largest = float(np.sqrt(max(lam_max, 0.0)))
        smallest = float(np.sqrt(max(lam_min, 0.0)))
        return dict(
            full_rank=bool(smallest > largest * 1e-7),
            smallest_singular_value=smallest,
            largest_singular_value=largest,
            condition_number=float(largest / max(smallest, 1e-300)),
            method="shift_invert_normal_matrix",
        )
    return dict(
        full_rank=bool(smallest > largest * 1e-10),
        smallest_singular_value=float(smallest),
        largest_singular_value=float(largest),
        condition_number=float(largest / max(smallest, 1e-300)),
    )


def solve(graph):
    state = graph.initial.copy()
    settings = graph.config["optimization"]
    result = Result(state, "maximum_iterations")
    try:
        cost, r, j = graph.linearize(state)
    except ValueError as exc:
        result.status = f"invalid_initial_geometry: {exc}"
        return result
    result.conditioning = rank_diagnostics(j)
    result.history.append(dict(iteration=0, cost=cost, accepted=True, alpha=0.0))
    # Weakly determined directions (3 Nov D4/D5: a few camera poses between 1 Hz PX4 samples) used
    # to stop the solve before it started, so every tag kept its starting value.  Solve anyway
    # with a little damping instead: those directions then stay where they started, everything
    # the measurements do determine (the tags) is optimised.
    weak = not result.conditioning["full_rank"]
    quiet = 0
    damp = float(settings.get("weak_direction_damping", 5e-2)) if weak else 0.0
    for iteration in range(1, settings["maximum_iterations"] + 1):
        gradient = float(np.max(np.abs(j.T @ r)))
        if gradient < settings["gradient_infinity_tolerance"]:
            result.status = "converged_gradient"
            break
        # Right preconditioning changes only the units of the linear unknowns.
        # Pixel, angle and metre columns otherwise differ enough for LSMR to
        # exhaust its iterations on moderate, full-rank real flight graphs.
        column_norm = np.sqrt(np.asarray(j.power(2).sum(axis=0)).ravel())
        column_scale = 1.0 / np.maximum(column_norm, 1e-15)
        linear = lsmr(
            j.multiply(column_scale).tocsr(),
            -r,
            damp=damp,
            atol=1e-11,
            btol=1e-11,
            maxiter=max(100, 10 * graph.dimension),
        )
        delta = column_scale * linear[0]
        # 6 Nov (D13, D18): istop 7 = LSMR hit its iteration limit, 3 = its condition-number limit.  That
        # step is still a usable descent step (the line search below only takes it if the cost drops), so
        # only a non-finite step is a failure - before, the whole solve stopped at iteration 0.
        if linear[1] not in (0, 1, 2, 3, 4, 5, 7) or not np.isfinite(delta).all():
            result.status = "linear_solve_failed"
            break
        line = settings["line_search"]
        alpha = line["initial_alpha"]
        accepted = False
        while alpha >= line["minimum_alpha"]:
            trial = graph.retract(state, alpha * delta)
            try:
                newcost, newr, newj = graph.linearize(trial)
                accepted = np.isfinite(newcost) and newcost < cost
                reason = "cost_did_not_decrease" if not accepted else ""
            except ValueError as exc:
                newcost = None
                reason = str(exc)
            result.history.append(
                dict(
                    iteration=iteration,
                    alpha=alpha,
                    cost=newcost,
                    accepted=bool(accepted),
                    gradient_infinity=gradient,
                    linear_stop=linear[1],
                    reason=reason,
                )
            )
            if accepted:
                break
            alpha *= line["shrink_factor"]
        if not accepted:
            result.status = "no_valid_descent_step"
            break
        oldcost = cost
        state, cost, r, j = trial, newcost, newr, newj
        result.state = state
        max_rotation, max_translation, max_tag_move = 0.0, 0.0, 0.0
        for key, sl in graph.slices.items():
            d = alpha * delta[sl]
            if key[0] == "pose":
                max_rotation = max(max_rotation, np.linalg.norm(d[:3]))
                max_translation = max(max_translation, np.linalg.norm(d[3:]))
            else:
                max_rotation = max(max_rotation, abs(d[2]))
                max_translation = max(max_translation, np.linalg.norm(d[:2]))
                max_tag_move = max(max_tag_move, np.linalg.norm(d[:2]))
        # 5 Nov: the tags (what we report) stop moving long before the poses do (D6: ~30 of 100
        # iterations, 12 min).  Stop when no tag moved more than tag_step_tolerance_m for 3 steps.
        quiet = quiet + 1 if max_tag_move < float(settings.get("tag_step_tolerance_m", 0.002)) else 0
        if quiet >= 3:
            result.status = "converged_tags_settled"
            break
        if (
            max_rotation < settings["rotation_step_tolerance_rad"]
            and max_translation < settings["translation_step_tolerance_m"]
        ):
            result.status = "converged_step"
            break
        if (oldcost - cost) / max(1.0, oldcost) < settings["relative_cost_tolerance"]:
            result.status = "converged_cost"
            break
    result.conditioning = rank_diagnostics(j)
    if not result.conditioning["full_rank"]:
        if weak and result.status.startswith("converged"):
            result.status += "_damped_weak_directions"
            result.conditioning["note"] = ("some pose directions are not determined by the data; "
                                           "they were held near their start (damped solve)")
        else:
            result.status = "unobservable_rank_deficient"
    elif result.status.startswith("converged") and graph.dimension <= 600:
        # Local IRLS curvature only; not an independent calibrated posterior.
        _, s, vh = np.linalg.svd(j.toarray(), full_matrices=False)
        result.covariance = (vh.T / (s * s)) @ vh
    return result
