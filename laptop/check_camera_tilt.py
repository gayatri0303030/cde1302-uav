"""Measure the camera's REAL tilt from a flight's tag sightings and compare it with the mount file.

  python check_camera_tilt.py                         the newest run in logs\\
  python check_camera_tilt.py logs\\E1a_E1_ellipse_20261102_103000

How: each clean sighting (tag 0-3, hamming 0) has its 4 corner pixels.  Trace those 4 rays down to
the floor at the drone's height (PX4 z and roll/pitch at that moment) for a trial tilt.  Only the
true tilt makes the tag come out 0.12 m wide on the floor.  The median over all sightings is the
measured tilt.  Same method that confirmed 65 deg on 30 Oct.  run_baseline.py prints this after
every flight too.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FX, FY, CX, CY = 183.29, 182.86, 161.04, 120.12          # 320x240 calibration (26 Oct)
TAG_M = 0.12


def _q2r(qx, qy, qz, qw):
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def _rbc(p):
    s, c = math.sin(p), math.cos(p)
    return np.array([[0.0, -s, c], [1.0, 0.0, 0.0], [0.0, c, s]])


def measure(run: Path, fwd: float, down: float, ids=("0", "1", "2", "3")):
    """Median tilt (deg below horizontal) and per-sighting values, from raw_detections.csv."""
    path = run / "raw_detections.csv"
    if not path.exists():
        return None, []
    t = np.array([fwd, 0.0, down])
    vals = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("tag_id") not in ids or r.get("hamming_distance") != "0" or not r.get("px4_qw"):
                continue
            try:
                ra = _q2r(*(float(r[k]) for k in ("px4_qx", "px4_qy", "px4_qz", "px4_qw")))
                h = -float(r["px4_z_m"])
                corners = [(float(r[f"corner_{i}_u"]), float(r[f"corner_{i}_v"])) for i in range(4)]
            except (KeyError, ValueError):
                continue
            if h < 0.2:                                      # on the ground / taking off: skip
                continue
            yaw = math.atan2(ra[1, 0], ra[0, 0])
            cz, sz = math.cos(-yaw), math.sin(-yaw)
            ra = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ ra   # roll/pitch only
            cam = ra @ t
            best = None
            for k in range(50, 851, 2):
                p = math.radians(k / 10)
                pts = []
                for u, v in corners:
                    d = ra @ _rbc(p) @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
                    if d[2] <= 1e-3:
                        pts = None
                        break
                    pts.append(cam + (h - cam[2]) / d[2] * d)
                if pts is None:
                    continue
                side = np.mean([np.linalg.norm(pts[i][:2] - pts[(i + 1) % 4][:2]) for i in range(4)])
                e = abs(side - TAG_M)
                if best is None or e < best[0]:
                    best = (e, k / 10)
            if best is not None and best[0] < 0.01:
                vals.append(best[1])
    if not vals:
        return None, []
    vals.sort()
    return vals[len(vals) // 2], vals


def configured(run: Path):
    """(pitch, fwd, down) the run used, from its mapper_config_used.yaml."""
    import yaml
    c = yaml.safe_load((run / "mapper_config_used.yaml").read_text())
    ext = c["camera"]["extrinsic_T_B_C"]
    R = ext["rotation_matrix"]
    pitch = math.degrees(math.atan2(float(R[2][2]), float(R[0][2])))
    t = [float(v) for v in ext["translation_m"]]
    return pitch, t[0], t[2]


def report(run: Path, out=print) -> float | None:
    try:
        pitch, fwd, down = configured(run)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    med, vals = measure(run, fwd, down)
    if med is None or len(vals) < 3:
        out(f"camera tilt check: not enough clean tag sightings ({len(vals)}) to measure the tilt")
        return None
    q1, q3 = vals[len(vals) // 4], vals[3 * len(vals) // 4]
    diff = med - pitch
    line = (f"camera tilt check: the tags measure the lens at {med:.0f} deg below horizontal "
            f"({len(vals)} sightings, middle half {q1:.0f}-{q3:.0f}); the mount file says {pitch:.0f} deg")
    if abs(diff) <= 5:
        out(line + " - OK")
    else:
        out(line + f" - MISMATCH by {diff:+.0f} deg.  Tag positions from this run are off.  Either re-tilt "
                   f"the camera to {pitch:.0f} deg, or tell the code the real angle: "
                   f"python set_camera_mount.py --pitch {med:.0f}")
    return med


def main() -> int:
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = sorted((p for p in (HERE / "logs").glob("*") if (p / "raw_detections.csv").exists()),
                      key=lambda p: p.stat().st_mtime)
        if not runs:
            print("no runs in logs\\")
            return 1
        run = runs[-1]
    print(run.name)
    report(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
