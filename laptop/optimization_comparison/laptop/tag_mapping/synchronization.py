"""Clock mapping and independently bracketed position/attitude interpolation."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import unverified
from .geometry import pose_from_quaternion, transform


def telemetry_streams(events, c):
    """Select one source. Legacy assumptions require the explicit replay flag."""
    position = [e for e in events if e["kind"] == "position"]
    attitude = [e for e in events if e["kind"] == "attitude"]
    if c["mavlink"]["primary_pose_source"] == "ODOMETRY":
        position, attitude = [], []
        for e in [e for e in events if e["kind"] == "odometry"]:
            if e["frame_id"] != 1 or e["child_frame_id"] != 12:
                raise ValueError("ODOMETRY requires LOCAL_NED/BODY_FRD frames")
            p, q = e["position"], e["quaternion_xyzw"]
            pose_from_quaternion(p, q)
            position.append(dict(e, x=p[0], y=p[1], z=p[2]))
            attitude.append(dict(e, qx=q[0], qy=q[1], qz=q[2], qw=q[3]))
    elif c["legacy_replay"]["allow_assumed_pose"]:
        legacy = [e for e in events if e["kind"] == "legacy_pose"]
        if legacy:
            if position or attitude:
                raise ValueError(
                    "select either legacy_pose or full telemetry, not both"
                )
            if not c["frames"]["start"][
                "input_positions_already_ground_start_relative"
            ]:
                raise ValueError(
                    "legacy yaw-only inputs require explicitly confirmed ground-relative coordinates"
                )
            settings = c["legacy_replay"]
            for e in legacy:
                if "yaw_rad" not in e:
                    raise ValueError("legacy yaw is missing")
                rpy = [
                    settings["assumed_roll_rad"],
                    settings["assumed_pitch_rad"],
                    e["yaw_rad"],
                ]
                if not np.isfinite(rpy).all():
                    raise ValueError("nonfinite legacy attitude")
                q = Rotation.from_euler("xyz", rpy).as_quat()
                position.append(
                    dict(e, z=e.get("z", settings["assumed_drone_world_z_m"]))
                )
                attitude.append(dict(e, qx=q[0], qy=q[1], qz=q[2], qw=q[3]))
    return position, attitude


def integer_ns(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} requires nonnegative integer nanoseconds")
    return value


class ClockMap:
    def __init__(
        self,
        scale,
        offset_ns,
        capture_offset_ns=0,
        sigma_ns=0,
        origin_ns=0,
        target_origin_ns=0,
    ):
        if (
            not np.isfinite([scale, offset_ns, capture_offset_ns, sigma_ns]).all()
            or scale <= 0
            or sigma_ns < 0
        ):
            raise ValueError("invalid clock mapping")
        self.scale, self.offset_ns = scale, offset_ns
        self.capture_offset_ns, self.sigma_ns = capture_offset_ns, sigma_ns
        self.origin_ns, self.target_origin_ns = origin_ns, target_origin_ns

    def map(self, ns):
        integer_ns(ns, "camera time")
        delta = ns - self.origin_ns
        offset = round(self.offset_ns)
        capture = round(self.capture_offset_ns)
        return (
            self.target_origin_ns
            + delta
            + offset
            + capture
            + round(
                (self.scale - 1) * delta
                + (self.offset_ns - offset)
                + (self.capture_offset_ns - capture)
            )
        )

    @classmethod
    def fit(cls, samples, capture_offset_ns=0):
        if len(samples) < 3:
            raise ValueError("clock fitting needs at least three measured pairs")
        x0, y0 = samples[0]["camera_time_ns"], samples[0]["fc_time_ns"]
        integer_ns(x0, "camera time")
        integer_ns(y0, "FC time")
        x = np.array(
            [integer_ns(s["camera_time_ns"], "camera time") - x0 for s in samples],
            float,
        )
        y = np.array(
            [integer_ns(s["fc_time_ns"], "FC time") - y0 for s in samples], float
        )
        sigmas = np.array([s["sigma_ns"] for s in samples], float)
        if np.ptp(x) == 0 or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("clock pairs need time span and positive uncertainties")
        # Seconds keep the two design columns similarly scaled, even at epoch time.
        a = np.c_[x / 1e9, np.ones(len(x))]
        coef = np.linalg.lstsq(a / sigmas[:, None], y / sigmas, rcond=None)[0]
        err = y - a @ coef
        if np.max(np.abs(err) / sigmas) > 5:
            raise ValueError("inconsistent clock measurements (>5 sigma)")
        return cls(
            coef[0] / 1e9,
            coef[1],
            capture_offset_ns,
            float(max(np.sqrt(np.mean(err**2)), np.max(sigmas))),
            x0,
            y0,
        )


def bracket(samples, time_ns, max_gap_ns, time_key="fc_time_ns"):
    ordered = sorted(samples, key=lambda s: s[time_key])
    times = [integer_ns(s[time_key], time_key) for s in ordered]
    if len(set(times)) != len(times):
        # Identical repeated publications are fine; conflicting same-time data are not.
        unique = {}
        for item in ordered:
            t = item[time_key]
            payload = {
                k: v
                for k, v in item.items()
                if k not in ("sequence", "host_receive_ns")
            }
            if t in unique and payload != unique[t][1]:
                raise ValueError("conflicting telemetry at identical timestamp")
            unique[t] = item, payload
        ordered = [unique[t][0] for t in sorted(unique)]
        times = sorted(unique)
    i = bisect_left(times, time_ns)
    if i < len(times) and times[i] == time_ns:
        return ordered[i], ordered[i], 0.0
    if i == 0 or i == len(times):
        raise ValueError("missing telemetry bracket; extrapolation unsupported")
    gap = times[i] - times[i - 1]
    if gap > max_gap_ns:
        raise ValueError(f"telemetry gap {gap/1e6:g} ms exceeds maximum")
    return ordered[i - 1], ordered[i], (time_ns - times[i - 1]) / gap


def interpolate_pose(position, attitude, ns, max_gap_ns, time_key="fc_time_ns"):
    a, b, f = bracket(position, ns, max_gap_ns, time_key)
    p = (1 - f) * np.array([a[k] for k in ("x", "y", "z")]) + f * np.array(
        [b[k] for k in ("x", "y", "z")]
    )
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("missing/nonfinite telemetry position")
    a, b, f = bracket(attitude, ns, max_gap_ns, time_key)
    q = [np.array([s[k] for k in ("qx", "qy", "qz", "qw")]) for s in (a, b)]
    for quat in q:
        pose_from_quaternion([0, 0, 0], quat)
    r = Slerp([0, 1], Rotation.from_quat(q))([f]).as_matrix()[0]
    return transform(r, p)


def ground_reference(start, c):
    if start.get("coordinate_convention") != "NED" or not start.get("datum_source"):
        raise ValueError("start event needs coordinate_convention=NED and datum_source")
    integer_ns(start.get("fc_time_ns"), "start FC time")
    native = pose_from_quaternion(
        start.get("native_position_m"), start.get("quaternion_xyzw")
    )
    settings = c["frames"]["start"]
    origin = start.get(
        "native_ground_origin_position_m", settings["native_ground_origin_position_m"]
    )
    height = start.get(
        "initial_body_height_above_floor_m",
        settings["initial_body_height_above_floor_m"],
    )
    sigma = start.get(
        "initial_body_height_sigma_m", settings["initial_body_height_sigma_m"]
    )
    if height is not None and (
        type(height) not in (float, int) or not np.isfinite(height) or height <= 0
    ):
        raise ValueError("invalid measured start height")
    if settings["input_positions_already_ground_start_relative"]:
        origin = [0.0, 0.0, 0.0]
        if height is None:
            height = -native[2, 3]
    elif origin is None:
        if height is None or not settings["ground_origin_directly_below_initial_body"]:
            raise ValueError(
                "missing ground datum: supply surveyed origin or measured initial height"
            )
        origin = native[:3, 3] + [0, 0, height]
    origin = np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid ground origin")
    measured_height = origin[2] - native[2, 3]
    if height is not None and abs(measured_height - height) > 1e-5:
        raise ValueError("inconsistent start height and ground origin")
    if (
        type(sigma) not in (float, int)
        or not np.isfinite(sigma)
        or sigma <= 0
        or measured_height <= 0
    ):
        raise ValueError(
            "start needs positive measured height and finite positive height sigma"
        )
    world = native.copy()
    world[:3, 3] -= origin
    return origin, world, float(sigma)


def synchronize(events, run, c, frame_metadata=None, start_event=None):
    """Return immutable synchronized poses; rejected frames remain in diagnostics."""
    start_path = Path(run) / c["frames"]["start"]["event_file"]
    start = (
        dict(start_event)
        if start_event is not None
        else json.loads(start_path.read_text())
    )
    warnings = unverified(c)
    legacy = c["legacy_replay"]["allow_assumed_pose"]
    failures = []
    try:
        origin, start_pose, height_sigma = ground_reference(start, c)
    except (ValueError, TypeError) as exc:
        failures.append(str(exc))
    if warnings and not c["allow_unverified_measurements"]:
        failures.append("unverified measurements: " + ", ".join(warnings))
    run_ids = {e["run_id"] for e in events}
    if start.get("run_id") not in run_ids:
        failures.append("start event run ID mismatch")
    if failures:
        raise ValueError("; ".join(failures))
    metadata = {}
    for m in frame_metadata or []:
        key = (m["camera_id"], m["boot_nonce"], m["frame_id"])
        if key in metadata:
            raise ValueError("duplicate frame metadata")
        metadata[key] = m
    timing = c["timing"]
    groups = sorted({e.get("segment_id", 0) for e in events if e["kind"] == "frame"})
    result, diagnostics = [], []
    frame_identities = set()
    for segment in groups:
        subset = [e for e in events if e.get("segment_id", 0) == segment]
        position, attitude = telemetry_streams(subset, c)
        samples = [e for e in subset if e["kind"] == "clock_samples"]
        clock = None
        if timing["capture_timestamp_offset_ns"] is not None:
            unscoped_samples = [s for s in samples if s.get("boot_nonce") is None]
            if unscoped_samples:
                clock = ClockMap.fit(
                    unscoped_samples, timing["capture_timestamp_offset_ns"]
                )
            elif all(
                timing[k] is not None for k in ("scale", "offset_ns", "sync_sigma_ns")
            ):
                clock = ClockMap(
                    timing["scale"],
                    timing["offset_ns"],
                    timing["capture_timestamp_offset_ns"],
                    timing["sync_sigma_ns"],
                )
        for original in [e for e in subset if e["kind"] == "frame"]:
            frame = dict(original)
            key = (frame.get("camera_id"), frame.get("boot_nonce"), frame["frame_id"])
            if key in metadata:
                m = metadata[key]
                if m.get("run_id") != original["run_id"]:
                    raise ValueError("frame metadata run ID mismatch")
                for field in (
                    "capture_time_ns",
                    "clock_domain",
                    "capture_timestamp_reference",
                    "exposure_verified",
                    "segment_id",
                ):
                    if field in m:
                        frame[field] = m[field]
                if frame.get("segment_id", 0) != segment:
                    raise ValueError("frame metadata segment mismatch")
            row = dict(
                frame_id=frame["frame_id"],
                segment_id=segment,
                camera_id=frame.get("camera_id"),
                fc_time_ns=None,
                status="rejected",
                reason="",
            )
            try:
                frame_identity = (segment,) + key
                if frame_identity in frame_identities:
                    raise ValueError(
                        "duplicate frame identity; measurements not counted twice"
                    )
                frame_identities.add(frame_identity)
                if frame.get("status") == "rejected":
                    raise ValueError(
                        frame.get("reason", "frame rejected at collection")
                    )
                if frame.get("camera_id") != c["camera"]["camera_id"]:
                    raise ValueError("frame camera ID does not match calibration")
                if (
                    frame["width"] != c["camera"]["width"]
                    or frame["height"] != c["camera"]["height"]
                    or frame["calibration_id"] != c["camera"]["calibration_id"]
                ):
                    raise ValueError("frame/calibration geometry mismatch")
                approximate = bool(warnings) or legacy
                if (
                    frame.get("exposure_verified")
                    and frame.get("capture_timestamp_reference")
                    == timing["capture_timestamp_reference"]
                ):
                    ns = integer_ns(frame.get("capture_time_ns"), "exposure time")
                    domain = frame.get("clock_domain")
                    if domain == timing["telemetry_clock_domain"]:
                        pass
                    elif domain == timing["camera_clock_domain"]:
                        boot_samples = [
                            s
                            for s in samples
                            if s.get("boot_nonce") == frame.get("boot_nonce")
                        ]
                        if (
                            boot_samples
                            and timing["capture_timestamp_offset_ns"] is not None
                        ):
                            frame_clock = ClockMap.fit(
                                boot_samples, timing["capture_timestamp_offset_ns"]
                            )
                        elif frame.get("camera_clock_reset"):
                            raise ValueError(
                                "camera reboot requires new boot-specific clock measurements"
                            )
                        else:
                            frame_clock = clock
                        if frame_clock is None:
                            raise ValueError("unknown clock mapping")
                        ns = frame_clock.map(ns)
                        row.update(
                            clock_sigma_ns=frame_clock.sigma_ns,
                            clock_scale=frame_clock.scale,
                            clock_offset_ns=frame_clock.offset_ns,
                            clock_camera_reference_ns=frame_clock.origin_ns,
                            clock_fc_reference_ns=frame_clock.target_origin_ns,
                        )
                    else:
                        raise ValueError("unknown clock mapping")
                    time_key = "fc_time_ns"
                elif legacy:
                    ns = integer_ns(frame.get("host_receive_ns"), "legacy receive time")
                    time_key = "host_receive_ns"
                    approximate = True
                else:
                    raise ValueError(
                        "missing verified exposure timestamp; ECAM frame age is insufficient"
                    )
                native = interpolate_pose(
                    position,
                    attitude,
                    ns,
                    round(timing["max_bracket_gap_ms"] * 1e6),
                    time_key,
                )
                world = native.copy()
                world[:3, 3] -= origin
                row.update(
                    status="approximate" if approximate else "synchronized",
                    fc_time_ns=ns if time_key == "fc_time_ns" else None,
                    source_time_ns=ns,
                    time_key=time_key,
                    x=world[0, 3],
                    y=world[1, 3],
                    z=world[2, 3],
                    quaternion_xyzw=Rotation.from_matrix(world[:3, :3])
                    .as_quat()
                    .tolist(),
                )
                result.append(
                    dict(
                        frame=frame,
                        pose=world,
                        native_pose=native,
                        time_ns=ns,
                        segment=segment,
                        approximate=approximate,
                        time_key=time_key,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                row["reason"] = str(exc)
            diagnostics.append(row)
    return (
        result,
        diagnostics,
        dict(
            start=start,
            origin=origin,
            start_pose=start_pose,
            height_sigma=height_sigma,
            warnings=warnings,
            position_events=events,
        ),
    )
