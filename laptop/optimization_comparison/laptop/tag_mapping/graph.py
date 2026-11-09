"""Planar landmarks, immutable odometry measurements and sparse factors."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import sparse

from .config import identity, tag_size
from .geometry import (
    adjoint,
    ground_intersections,
    initialize_tag,
    inverse,
    log_right_jacobian,
    pixel_factor,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
    wrap,
)
from .synchronization import interpolate_pose, telemetry_streams


@dataclass
class State:
    poses: list
    tags: dict

    def copy(self):
        return State(
            [p.copy() for p in self.poses], {k: v.copy() for k, v in self.tags.items()}
        )


@dataclass
class Factor:
    kind: str
    keys: tuple
    measurement: object
    covariance: np.ndarray
    metadata: dict

    def evaluate(self, state, c):
        if self.kind == "pixel":
            i, key = self.keys
            r, jp, jt = pixel_factor(
                state.poses[i],
                state.tags[key],
                tag_size(c, key),
                self.measurement,
                c,
                self.metadata["convention"],
            )
            return r, {("pose", i): jp, ("tag", key): jt}
        if self.kind == "motion":
            i, j = self.keys
            relative = inverse(state.poses[i]) @ state.poses[j]
            r = se3_log(inverse(self.measurement) @ relative)
            jr = log_right_jacobian(r)
            return r, {("pose", i): -jr @ adjoint(inverse(relative)), ("pose", j): jr}
        if self.kind == "anchor":
            (key,) = self.keys
            return state.tags[key][:2] - self.measurement, {
                ("tag", key): np.c_[np.eye(2), np.zeros(2)]
            }
        if self.kind == "yaw":
            (key,) = self.keys
            return np.array([wrap(state.tags[key][2] - self.measurement)]), {
                ("tag", key): np.array([[0.0, 0.0, 1.0]])
            }
        if self.kind == "height":
            (i,) = self.keys
            j = np.r_[np.zeros(3), state.poses[i][2, :3]][None, :]
            return np.array([state.poses[i][2, 3] - self.measurement]), {("pose", i): j}
        if self.kind == "attitude":
            (i,) = self.keys
            r = so3_log(self.measurement.T @ state.poses[i][:3, :3])
            jr = log_right_jacobian(np.r_[r, [0, 0, 0]])[:3, :3]
            return r, {("pose", i): np.c_[jr, np.zeros((3, 3))]}
        raise ValueError(f"unknown factor {self.kind}")


class Graph:
    def __init__(self, c, state, pose_records, factors, diagnostics):
        self.config, self.initial, self.pose_records, self.factors, self.diagnostics = (
            c,
            state,
            pose_records,
            factors,
            diagnostics,
        )
        self.slices = {}
        offset = 0
        for i in range(len(state.poses)):
            n = 4 if i == 0 else 6
            self.slices["pose", i] = slice(offset, offset + n)
            offset += n
        for key in sorted(state.tags):
            self.slices["tag", key] = slice(offset, offset + 3)
            offset += 3
        self.dimension = offset

    def lift(self, state, i):
        if i != 0:
            return np.eye(6)
        j = np.zeros((6, 4))
        j[:3, :3] = np.eye(3)
        j[3:, 3] = state.poses[0][:3, :3].T @ np.array([0, 0, 1.0])
        return j

    def retract(self, state, delta):
        trial = state.copy()
        for i, p in enumerate(state.poses):
            d = delta[self.slices["pose", i]]
            if i == 0:
                # Reduced coordinate chart: fixed world XY, right rotation, free Z.
                # Equivalent to a right SE(3) update with translation lift
                # J_l(phi)^-1 R.T [0,0,dz]. Its derivative is lift().
                trial.poses[i][:3, :3] = p[:3, :3] @ so3_exp(d[:3])
                trial.poses[i][2, 3] = p[2, 3] + d[3]
            else:
                trial.poses[i] = p @ se3_exp(d)
        for key, t in state.tags.items():
            trial.tags[key] = t + delta[self.slices["tag", key]]
            trial.tags[key][2] = wrap(trial.tags[key][2])
        return trial

    def linearize(self, state, robust=True):
        residuals, row_indices, col_indices, entries = [], [], [], []
        cost = 0.0
        row = 0
        for factor in self.factors:
            r, blocks = factor.evaluate(state, self.config)
            l = np.linalg.cholesky(factor.covariance)
            r = np.linalg.solve(l, r)
            norm = np.linalg.norm(r)
            huber = (
                robust
                and factor.kind in self.config["optimization"]["robustify_factor_types"]
            )
            d = self.config["optimization"]["huber_delta_whitened"]
            weight = min(1.0, d / max(norm, 1e-30)) if huber else 1.0
            cost += d * (norm - 0.5 * d) if huber and norm > d else 0.5 * norm**2
            root = np.sqrt(weight)
            residuals.extend(root * r)
            for key, block in blocks.items():
                if key[0] == "pose":
                    block = block @ self.lift(state, key[1])
                block = root * np.linalg.solve(l, block)
                rr, cc = np.indices(block.shape)
                row_indices.extend((rr + row).ravel())
                col_indices.extend((cc + self.slices[key].start).ravel())
                entries.extend(block.ravel())
            row += len(r)
        jac = sparse.csr_matrix(
            (entries, (row_indices, col_indices)), shape=(row, self.dimension)
        )
        return float(cost), np.array(residuals), jac

    def residual_summary(self, state):
        result = []
        for f in self.factors:
            r, _ = f.evaluate(state, self.config)
            result.append(
                dict(
                    kind=f.kind,
                    keys=str(f.keys),
                    residual=r.tolist(),
                    whitened_norm=float(
                        np.linalg.norm(
                            np.linalg.solve(np.linalg.cholesky(f.covariance), r)
                        )
                    ),
                    **f.metadata,
                )
            )
        return result


def detection_reason(d, c):
    key = identity(d)
    detector, gating = c["detector"], c["gating"]
    if (
        key[0] != detector["family_or_dictionary"]
        or key[1] not in detector["allowed_tag_ids"]
    ):
        return "outside_whitelist"
    if d.get("hamming") is None or d.get("margin") is None:
        return "missing_detection_quality"
    if (
        d["hamming"] > gating["maximum_hamming_distance"]
        or d["margin"] <= gating["minimum_apriltag_decision_margin"]
    ):
        return "quality_gate"
    p = np.asarray(d["corners"], float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return "invalid_corners"
    if (
        not cv2.isContourConvex(p.astype(np.float32))
        or abs(cv2.contourArea(p.astype(np.float32))) < gating["minimum_quad_area_px2"]
    ):
        return "degenerate_quad"
    return None


def build_graph(synchronized, context, c, require_anchor=True):
    if not synchronized:
        raise ValueError("no synchronized camera frames")
    start, origin = context["start"], context["origin"]
    segment = start.get("segment_id", 0)
    diagnostics, records = [], []
    start_time = start["fc_time_ns"]
    records.append(
        dict(
            pose=context["start_pose"],
            time_ns=start_time,
            segment=segment,
            role="start",
        )
    )
    valid = []
    for s in synchronized:
        if s["segment"] != segment:
            diagnostics.append(
                dict(
                    frame_id=s["frame"]["frame_id"],
                    reason="reset_segment_without_ground_reference",
                )
            )
            continue
        if s["time_key"] != "fc_time_ns":
            if start.get("host_receive_ns") is None:
                raise ValueError("approximate replay needs start host_receive_ns")
            records[0]["time_ns"] = start["host_receive_ns"]
        if s["time_ns"] < records[0]["time_ns"]:
            diagnostics.append(
                dict(frame_id=s["frame"]["frame_id"], reason="before_start_event")
            )
            continue
        valid.append(s)
    if not valid:
        raise ValueError("no frames in the start reference segment")
    time_key = valid[0]["time_key"]
    if any(s["time_key"] != time_key for s in valid):
        raise ValueError("cannot mix verified and arrival-time poses within a graph")
    events = [
        e for e in context["position_events"] if e.get("segment_id", 0) == segment
    ]
    position, attitude = telemetry_streams(events, c)
    last_support = records[0]["time_ns"]
    end = max([s["time_ns"] for s in valid] + [e[time_key] for e in position])
    maxgap = round(c["timing"]["max_bracket_gap_ms"] * 1e6)
    for e in sorted(position, key=lambda e: e[time_key]):
        ns = e[time_key]
        if (
            ns > end
            or ns <= last_support
            or ns - last_support
            < c["optimization"]["keyframes"]["minimum_interval_s"] * 1e9
        ):
            continue
        try:
            p = interpolate_pose(position, attitude, ns, maxgap, time_key)
            p[:3, 3] -= origin
            records.append(
                dict(pose=p, time_ns=ns, segment=segment, role="motion_support")
            )
            last_support = ns
        except ValueError as exc:
            diagnostics.append(dict(time_ns=ns, reason=str(exc)))
    # Include the final telemetry state even when the run ends without tags.
    if position and end > last_support:
        try:
            p = interpolate_pose(position, attitude, end, maxgap, time_key)
            p[:3, 3] -= origin
            records.append(
                dict(pose=p, time_ns=end, segment=segment, role="motion_support")
            )
        except ValueError as exc:
            diagnostics.append(dict(time_ns=end, reason=str(exc)))
    for s in valid:
        records.append(
            dict(pose=s["pose"], time_ns=s["time_ns"], segment=segment, role="camera")
        )
    by_time = {r["time_ns"]: r for r in records[1:]}
    by_time[records[0]["time_ns"]] = records[0]
    records = [by_time[t] for t in sorted(by_time)]
    index = {r["time_ns"]: i for i, r in enumerate(records)}
    factors, guesses = [], {}
    for s in valid:
        counts = {}
        for d in s["frame"].get("detections", []):
            key = identity(d)
            counts[key] = counts.get(key, 0) + 1
        for d in s["frame"].get("detections", []):
            key = identity(d)
            meta = dict(
                frame_id=s["frame"]["frame_id"],
                family_or_dictionary=key[0],
                tag_id=key[1],
            )
            try:
                reason = detection_reason(d, c)
                if counts[key] > 1:
                    reason = "duplicate_ID_in_frame"
                if reason:
                    raise ValueError(reason)
                convention = s["frame"]["pixel_convention"]
                corners = ground_intersections(d["corners"], s["pose"], c, convention)
                guess = initialize_tag(corners, tag_size(c, key))
                guesses.setdefault(key, []).append(guess)
                diagnostics.append(
                    dict(
                        meta,
                        status="initialized",
                        ground_corners=corners.tolist(),
                        inferred_edge_lengths=np.linalg.norm(
                            np.roll(corners, -1, axis=0) - corners, axis=1
                        ).tolist(),
                    )
                )
                factors.append(
                    Factor(
                        "pixel",
                        (index[s["time_ns"]], key),
                        np.array(d["corners"]),
                        np.eye(8) * c["uncertainty"]["corner_sigma_px"] ** 2,
                        dict(meta, convention=convention),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics.append(dict(meta, status="rejected", reason=str(exc)))
    if not guesses:
        raise ValueError("no valid tag observations")
    tags = {}
    for key, values in guesses.items():
        values = np.array(values)
        # Medoid rejects inconsistent views without seeding from withheld survey.
        distances = np.linalg.norm(values[:, None, :2] - values[None, :, :2], axis=2)
        tags[key] = values[np.argmin(np.median(distances, axis=1))].copy()
        diagnostics.append(
            dict(
                family_or_dictionary=key[0],
                tag_id=key[1],
                status="initialization_spread",
                maximum_initial_spread_m=float(np.max(distances)),
                reason="Review spread for reused IDs, motion drift, calibration or timing error",
            )
        )
    anchors = [a for a in c["known_tags"] if a["enabled"]]
    observed = []
    for a in anchors:
        key = identity(a)
        if key not in tags:
            if require_anchor:
                raise ValueError(f"enabled anchor {key} unobserved")
            diagnostics.append(
                dict(reason=f"enabled anchor {key} unobserved", status="unused_anchor")
            )
            continue
        tags[key][:2] = a["position_m"][:2]
        factors.append(
            Factor(
                "anchor",
                (key,),
                np.array(a["position_m"][:2]),
                np.diag(np.square(a["position_sigma_xy_m"])),
                {},
            )
        )
        observed.append(key)
        if a["surveyed_yaw_rad"] is not None:
            tags[key][2] = a["surveyed_yaw_rad"]
            factors.append(
                Factor(
                    "yaw",
                    (key,),
                    a["surveyed_yaw_rad"],
                    np.array([[a["surveyed_yaw_sigma_rad"] ** 2]]),
                    {},
                )
            )
    o, u = c["optimization"], c["uncertainty"]
    if require_anchor and not observed and not o["allow_anchor_free_debug_run"]:
        raise ValueError(
            "no observed enabled survey anchor; use explicit anchor-free debug mode"
        )
    for i in range(len(records) - 1):
        a, b = records[i], records[i + 1]
        continuous = True
        for stream in (position, attitude):
            times = sorted(
                {
                    e[time_key]
                    for e in stream
                    if a["time_ns"] - maxgap <= e[time_key] <= b["time_ns"] + maxgap
                }
            )
            inner = (
                [a["time_ns"]]
                + [t for t in times if a["time_ns"] < t < b["time_ns"]]
                + [b["time_ns"]]
            )
            if not times or max(np.diff(inner)) > maxgap:
                continuous = False
        if not continuous:
            diagnostics.append(
                dict(time_ns=a["time_ns"], reason="motion_edge_rejected_telemetry_gap")
            )
            continue
        measurement = inverse(a["pose"]) @ b["pose"]
        measurement.setflags(write=False)
        dt = (b["time_ns"] - a["time_ns"]) / 1e9
        sigma = np.r_[
            u["fallback_odometry_rotation_sigma_rad"],
            u["fallback_odometry_translation_sigma_m"],
        ] * np.sqrt(dt / u["fallback_sigma_reference_interval_s"])
        factors.append(
            Factor(
                "motion",
                (i, i + 1),
                measurement,
                np.diag(sigma * sigma),
                dict(dt_s=dt, covariance_source="heuristic_correlated_PX4"),
            )
        )
    factors.extend(
        [
            Factor(
                "height",
                (0,),
                records[0]["pose"][2, 3],
                np.array([[context["height_sigma"] ** 2]]),
                {},
            ),
            Factor(
                "attitude",
                (0,),
                records[0]["pose"][:3, :3].copy(),
                np.diag(np.square(u["start_attitude_prior_sigma_rad"])),
                {},
            ),
        ]
    )
    # A sighting that cannot be projected from the starting guess (a tag corner behind the camera,
    # e.g. a long-range decode at the edge of the picture) would stop the whole solve: drop it.
    kept_f = []
    for fct in factors:
        if fct.kind == "pixel":
            i, key = fct.keys
            try:
                pixel_factor(records[i]["pose"], tags[key], tag_size(c, key), fct.measurement, c,
                             fct.metadata["convention"])
            except ValueError as exc:
                diagnostics.append(dict(fct.metadata, status="rejected",
                                        reason=f"pixel_factor_invalid_at_start: {exc}"))
                continue
        kept_f.append(fct)
    factors = kept_f
    # Connected components include landmark bridges across detection-free periods.
    nodes = {("pose", i): set() for i in range(len(records))}
    nodes.update({("tag", k): set() for k in tags})
    for f in factors:
        if f.kind == "motion":
            a, b = [("pose", i) for i in f.keys]
        elif f.kind == "pixel":
            a, b = ("pose", f.keys[0]), ("tag", f.keys[1])
        else:
            continue
        nodes[a].add(b)
        nodes[b].add(a)
    visited = {("pose", 0)}
    pending = list(visited)
    while pending:
        for node in nodes[pending.pop()] - visited:
            visited.add(node)
            pending.append(node)
    if visited != set(nodes):
        # A pose cut off on BOTH sides by a PX4 telemetry gap that saw no tag carries no map
        # information (2 Nov E8a: one 1 Hz PX4 sample between two dropped ones, during take-off).
        # Drop such tag-free islands.  A cut-off piece that holds tags is still a real error.
        lost = set(nodes) - visited
        if any(kind == "tag" for kind, _ in lost):
            raise ValueError(
                "disconnected graph: telemetry gap/reset lacks a valid reference connection"
            )
        drop = {i for _, i in lost}
        keep = [i for i in range(len(records)) if i not in drop]
        renum = {old: new for new, old in enumerate(keep)}
        kept = []
        for f in factors:
            if f.kind == "motion":
                if f.keys[0] in drop or f.keys[1] in drop:
                    continue
                f = dataclasses.replace(f, keys=(renum[f.keys[0]], renum[f.keys[1]]))
            elif f.kind == "pixel":
                f = dataclasses.replace(f, keys=(renum[f.keys[0]], f.keys[1]))
            elif f.kind in ("height", "attitude"):
                f = dataclasses.replace(f, keys=(renum[f.keys[0]],))
            kept.append(f)
        for i in sorted(drop):
            diagnostics.append(
                dict(
                    time_ns=records[i]["time_ns"],
                    role=records[i]["role"],
                    reason="pose_dropped_isolated_by_telemetry_gaps_no_tags",
                )
            )
        records = [records[i] for i in keep]
        factors = kept
    if (
        u["add_height_prior_from_same_px4_source"]
        or u["add_attitude_prior_at_every_pose_from_same_px4_source"]
    ):
        raise ValueError("repeated correlated PX4 absolute priors unsupported")
    if (
        u["independent_height_sigma_m"] is not None
        or u["independent_attitude_sigma_rad"] is not None
    ):
        raise ValueError(
            "independent sensor priors require an implemented sensor adapter"
        )
    return Graph(
        c,
        State([r["pose"].copy() for r in records], tags),
        records,
        factors,
        diagnostics,
    )
