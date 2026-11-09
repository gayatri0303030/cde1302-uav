"""Add DERIVED estimates for the PX4 columns the drone does not send (no .ulg, no MAVLink stream).

  python derive_columns.py logs\\fly_<time>

Reads raw_detections.csv (+ px4_telemetry.csv from the same folder if present) and writes
raw_detections_derived.csv.  The raw columns are NOT touched; every estimate goes in a new column
ending in _derived, so nobody mistakes it for a PX4 measurement.

  px4_vx_mps_derived, px4_vy_mps_derived   central difference of PX4 x, y (10 Hz telemetry)
  px4_yaw_rate_radps_derived               central difference of PX4 yaw (unwrapped)
  px4_q{x,y,z,w}_derived                   from PX4 yaw ONLY, roll = pitch = 0 assumed (level hover)
  px4_z_m_derived                          -(camera height above the floor), from the tag pose:
                                           height = |tag normal . tag position| in the camera frame,
                                           median over the accepted tags in the same camera frame.
                                           Scales with tag.size_m - if the tag size is wrong, so is this.
  derived_dt_s                             time span of the two samples used for the differences

Not derivable here (left empty): px4_vz, roll rate, pitch rate.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

MAX_DT_S = 0.6   # refuse a difference over a gap longer than this


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def quat_to_R(qx, qy, qz, qw):
    return [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]


def load_series(folder: Path, dets: list[dict]):
    """(t, x, y, yaw) samples, from px4_telemetry.csv if present, else from the detections."""
    pts = {}
    tel = folder / "px4_telemetry.csv"
    if tel.exists():
        with open(tel, newline="") as fh:
            for r in csv.DictReader(fh):
                t = f(r.get("t_epoch_sample")) or f(r.get("t_epoch_rx"))
                x, y, yaw = f(r.get("px4_x_m")), f(r.get("px4_y_m")), f(r.get("px4_yaw_rad"))
                if None not in (t, x, y, yaw):
                    pts[round(t, 4)] = (t, x, y, yaw)
        src = "px4_telemetry.csv"
    if not pts:
        for r in dets:
            t, x, y, yaw = (f(r.get(k)) for k in ("px4_timestamp", "px4_x_m", "px4_y_m", "px4_yaw_rad"))
            if None not in (t, x, y, yaw):
                pts[round(t, 4)] = (t, x, y, yaw)
        src = "raw_detections.csv (no px4_telemetry.csv found)"
    s = sorted(pts.values())
    # unwrap yaw
    out, prev, off = [], None, 0.0
    for t, x, y, yaw in s:
        if prev is not None:
            d = yaw + off - prev
            if d > math.pi:
                off -= 2 * math.pi
            elif d < -math.pi:
                off += 2 * math.pi
        prev = yaw + off
        out.append((t, x, y, prev))
    return out, src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fly_dir", help="logs\\fly_<time> folder (or its raw_detections.csv)")
    args = ap.parse_args()
    raw = Path(args.fly_dir)
    if raw.is_dir():
        raw = raw / "raw_detections.csv"
    with open(raw, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = list(rd.fieldnames or [])
        dets = list(rd)

    series, src = load_series(raw.parent, dets)
    ts = [p[0] for p in series]
    print(f"PX4 x/y/yaw samples: {len(series)} from {src}")

    # camera height per camera frame, from accepted tag poses
    heights = defaultdict(list)
    for r in dets:
        t = [f(r.get(k)) for k in ("tag_tx_m", "tag_ty_m", "tag_tz_m")]
        q = [f(r.get(k)) for k in ("tag_qx", "tag_qy", "tag_qz", "tag_qw")]
        if None in t or None in q:
            continue
        R = quat_to_R(*q)
        n = (R[0][2], R[1][2], R[2][2])          # tag z-axis (floor normal) in the camera frame
        heights[r.get("frame_id")].append(abs(n[0] * t[0] + n[1] * t[1] + n[2] * t[2]))

    new = ["px4_vx_mps_derived", "px4_vy_mps_derived", "px4_yaw_rate_radps_derived",
           "px4_qx_derived", "px4_qy_derived", "px4_qz_derived", "px4_qw_derived",
           "px4_z_m_derived", "derived_dt_s"]
    nv = nz = 0
    for r in dets:
        t = f(r.get("camera_capture_timestamp")) or f(r.get("px4_timestamp"))
        yaw = f(r.get("px4_yaw_rad"))
        if yaw is not None:
            qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
            if qw < 0:
                qz, qw = -qz, -qw
            r.update(px4_qx_derived="0.000000", px4_qy_derived="0.000000",
                     px4_qz_derived=f"{qz:.6f}", px4_qw_derived=f"{qw:.6f}")
        if t is not None and len(series) >= 3:
            i = bisect.bisect_left(ts, t)
            i = min(max(i, 1), len(series) - 2)
            if abs(ts[i - 1] - t) < abs(ts[i] - t):
                i = max(i - 1, 1)
            a, b = series[i - 1], series[i + 1]
            dt = b[0] - a[0]
            if 0 < dt <= MAX_DT_S:
                r.update(px4_vx_mps_derived=f"{(b[1] - a[1]) / dt:.4f}",
                         px4_vy_mps_derived=f"{(b[2] - a[2]) / dt:.4f}",
                         px4_yaw_rate_radps_derived=f"{(b[3] - a[3]) / dt:.4f}",
                         derived_dt_s=f"{dt:.3f}")
                nv += 1
        h = heights.get(r.get("frame_id"))
        if h:
            r["px4_z_m_derived"] = f"{-median(h):.4f}"
            nz += 1

    out = raw.with_name("raw_detections_derived.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + [c for c in new if c not in cols])
        w.writeheader()
        w.writerows(dets)
    hs = [x for v in heights.values() for x in v]
    print(f"{len(dets)} rows -> {out}")
    print(f"  velocity / yaw rate filled: {nv}   z from tags filled: {nz}   quaternion (yaw-only): all with yaw")
    if hs:
        print(f"  camera height from tags: median {median(hs):.3f} m, range {min(hs):.3f}-{max(hs):.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
