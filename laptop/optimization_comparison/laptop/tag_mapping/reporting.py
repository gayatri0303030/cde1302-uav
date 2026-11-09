"""Separate fit diagnostics from independent survey evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import identity, read_yaml
from .data import dump_json, write_csv


def evaluate(initial, optimized, survey_path, anchor_keys):
    if not survey_path or not Path(survey_path).exists():
        return dict(status="independent_accuracy_not_evaluated")
    survey = read_yaml(survey_path)
    if set(survey) != {"tags"} or not isinstance(survey["tags"], list):
        raise ValueError("held-out survey requires a tags list")
    rows = []
    seen = set()
    for tag in survey["tags"]:
        key = identity(tag)
        if key in seen or key in anchor_keys:
            raise ValueError(
                "held-out survey duplicates an ID or overlaps fitting anchors"
            )
        seen.add(key)
        p = np.asarray(tag["position_m"], float)
        if (
            p.shape != (3,)
            or not np.isfinite(p).all()
            or p[2] != 0
            or tag.get("frame", "ground_start_NED") != "ground_start_NED"
        ):
            raise ValueError("invalid held-out NED ground survey")
        row = dict(
            family_or_dictionary=key[0], tag_id=key[1], survey_position_m=p.tolist()
        )
        if key not in optimized:
            row["status"] = "unobserved"
        else:
            row.update(
                status="evaluated",
                initial_error_m=float(np.linalg.norm(initial[key][:2] - p[:2])),
                optimized_error_m=float(np.linalg.norm(optimized[key][:2] - p[:2])),
            )
        rows.append(row)
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    out = dict(
        status="evaluated" if evaluated else "independent_accuracy_not_evaluated",
        tags=rows,
    )
    for kind in ("initial", "optimized"):
        errors = [r[kind + "_error_m"] for r in evaluated]
        if errors:
            out[kind] = dict(
                horizontal_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                median_m=float(np.median(errors)),
                maximum_m=float(max(errors)),
            )
    return out


def export_results(
    directory, graph, result, context, sync_rows, evaluation=None, provenance=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = result.state
    approximate = bool(context["warnings"]) or any(
        r["status"] == "approximate" for r in sync_rows
    )
    anchor_keys = {f.keys[0] for f in graph.factors if f.kind == "anchor"}
    tags = []
    for key, t in sorted(state.tags.items()):
        sl = graph.slices["tag", key]
        cov = (
            result.covariance[sl, sl].tolist()
            if result.covariance is not None
            else None
        )
        tags.append(
            dict(
                family_or_dictionary=key[0],
                tag_id=key[1],
                x_m=float(t[0]),
                y_m=float(t[1]),
                z_m=0.0,
                theta_rad=float(t[2]),
                initial_x_m=float(graph.initial.tags[key][0]),
                initial_y_m=float(graph.initial.tags[key][1]),
                initial_theta_rad=float(graph.initial.tags[key][2]),
                anchor=key in anchor_keys,
                covariance_xytheta=cov,
                status="approximate" if approximate else result.status,
                solver_status=result.status,
                z_variance=0.0,
                z_variance_reason="exact_assumed_ground_constraint",
            )
        )
    trajectory = []
    for i, p in enumerate(state.poses):
        q = Rotation.from_matrix(p[:3, :3]).as_quat()
        initial = graph.initial.poses[i]
        trajectory.append(
            dict(
                pose_index=i,
                time_ns=graph.pose_records[i]["time_ns"],
                segment_id=graph.pose_records[i]["segment"],
                role=graph.pose_records[i]["role"],
                x_m=float(p[0, 3]),
                y_m=float(p[1, 3]),
                z_m=float(p[2, 3]),
                initial_x_m=float(initial[0, 3]),
                initial_y_m=float(initial[1, 3]),
                initial_z_m=float(initial[2, 3]),
                qx=float(q[0]),
                qy=float(q[1]),
                qz=float(q[2]),
                qw=float(q[3]),
                attitude_shift_rad=float(
                    (
                        Rotation.from_matrix(initial[:3, :3]).inv()
                        * Rotation.from_matrix(p[:3, :3])
                    ).magnitude()
                ),
            )
        )
    initial_res = graph.residual_summary(graph.initial)
    final_res = graph.residual_summary(state)
    for record in final_res:
        if record["kind"] == "pixel":
            record["corner_euclidean_rms_px"] = float(
                np.sqrt(np.sum(np.square(record["residual"])) / 4)
            )
            record["above_diagnostic_threshold"] = (
                record["corner_euclidean_rms_px"]
                > graph.config["gating"]["maximum_corner_euclidean_rms_px"]
            )
    payload = dict(
        schema_version=1,
        solver=result.solver,
        status=result.status,
        approximate=approximate,
        anchor_status=(
            "anchor_corrected"
            if anchor_keys and result.status.startswith("converged")
            else "not_anchor_corrected"
        ),
        assumptions=context["warnings"],
        conditioning=result.conditioning,
        tags=tags,
        trajectory=trajectory,
        initial_residuals=initial_res,
        final_residuals=final_res,
        history=result.history,
        diagnostics=graph.diagnostics,
        synchronization=sync_rows,
        uncertainty_note="Local IRLS curvature with heuristic correlated PX4 weights; not independently calibrated. Null when unavailable.",
        evaluation=evaluation or dict(status="independent_accuracy_not_evaluated"),
        provenance=provenance,
    )
    dump_json(directory / "results.json", payload)
    write_csv(directory / "tags.csv", tags, list(tags[0]) if tags else ["tag_id"])
    write_csv(directory / "trajectory.csv", trajectory, list(trajectory[0]))
    write_csv(
        directory / "synchronized_poses.csv",
        sync_rows,
        [
            "frame_id",
            "camera_id",
            "segment_id",
            "fc_time_ns",
            "source_time_ns",
            "time_key",
            "status",
            "reason",
            "x",
            "y",
            "z",
            "quaternion_xyzw",
            "clock_scale",
            "clock_offset_ns",
            "clock_sigma_ns",
            "clock_camera_reference_ns",
            "clock_fc_reference_ns",
        ],
    )
    write_csv(
        directory / "residuals.csv",
        [
            dict(r, stage=stage)
            for stage, rr in [("initial", initial_res), ("optimized", final_res)]
            for r in rr
        ],
        [
            "stage",
            "kind",
            "keys",
            "frame_id",
            "family_or_dictionary",
            "tag_id",
            "residual",
            "whitened_norm",
            "corner_euclidean_rms_px",
            "above_diagnostic_threshold",
        ],
    )
    write_csv(
        directory / "iterations.csv",
        result.history,
        [
            "iteration",
            "cost",
            "accepted",
            "alpha",
            "gradient_infinity",
            "linear_stop",
            "reason",
        ],
    )
    write_csv(
        directory / "diagnostics.csv",
        graph.diagnostics,
        [
            "frame_id",
            "family_or_dictionary",
            "tag_id",
            "time_ns",
            "status",
            "reason",
            "ground_corners",
            "inferred_edge_lengths",
            "maximum_initial_spread_m",
        ],
    )
    write_csv(
        directory / "evaluation.csv",
        (evaluation or {}).get("tags", []),
        [
            "family_or_dictionary",
            "tag_id",
            "status",
            "survey_position_m",
            "initial_error_m",
            "optimized_error_m",
        ],
    )
    plot_map(directory / "map.png", graph, state, evaluation)
    return payload


def plot_map(path, graph, state, evaluation=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    for poses, tags, label, style in [
        (graph.initial.poses, graph.initial.tags, "initial", "--"),
        (state.poses, state.tags, "optimized", "-"),
    ]:
        p = np.array([p[:3, 3] for p in poses])
        (line,) = ax.plot(p[:, 1], p[:, 0], style, label=label + " trajectory")
        for key, t in tags.items():
            ax.plot(t[1], t[0], "o", color=line.get_color())
            ax.annotate(
                f"{key[0]}:{key[1]} ({label})",
                (t[1], t[0]),
                xytext=(8, 10 if label == "initial" else -14),
                textcoords="offset points",
                fontsize=8,
                color=line.get_color(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            )
    for a in graph.config["known_tags"]:
        if a["enabled"]:
            ax.plot(
                a["position_m"][1],
                a["position_m"][0],
                "k*",
                markersize=12,
                label=f'survey anchor {a["tag_id"]}',
            )
    for t in (evaluation or {}).get("tags", []):
        p = t["survey_position_m"]
        ax.plot(p[1], p[0], "g+", markersize=12, label=f'held-out survey {t["tag_id"]}')
    ax.set(
        xlabel="East (m)",
        ylabel="North (m)",
        title="Ground map — NED, floor Down = 0 m",
    )
    ax.axis("equal")
    ax.grid()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
