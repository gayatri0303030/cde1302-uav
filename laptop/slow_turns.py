"""Slow-turn version of a trajectory, so the nose (--yaw-follow) turns gently on ANY firmware.

  python slow_turns.py experiment_plans\\E4_ellipse_yaw.csv                 -> ..._slow.csv + .png
  python slow_turns.py experiment_plans\\E8_rectangle_yaw.csv --turn-dps 20

With --yaw-follow the firmware points the nose along the path, so the nose turns exactly as fast
as the path changes direction (on the first yaw-follow firmware up to ~115 deg/s at corners and at
the start).  This adds points: every sharp corner becomes a short smooth curve, and the drone slows
down wherever the path bends so the direction of travel turns at most --turn-dps (default 25).
Straights stay at --speed.  Same route (corners cut by a few cm), same start and end; ends with
the nose forward again.  Refused if longer than the drone's 120 s.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_trajectory as mt          # noqa: E402
import plan_coverage as pc            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--turn-dps", type=float, default=25.0, help="fastest the path (and nose) turns, deg/s")
    ap.add_argument("--speed", type=float, default=0.2, help="speed on the straights, m/s")
    ap.add_argument("--out", default=None, help="default: <name>_slow.csv beside the input")
    a = ap.parse_args()
    src = Path(a.csv)
    A = np.loadtxt(src, delimiter=",", skiprows=1)
    P = A[:, 1:3]
    Q = mt.slow_turns(P, turn_dps=a.turn_dps, speed=a.speed, end_hold=1.5)
    if len(Q) > mt.MAX_POINTS:
        print(f"not made: {len(Q) * pc.DT:.0f} s is over the drone's {mt.MAX_POINTS * pc.DT:.0f} s - "
              f"use a higher --turn-dps")
        return 1
    nose = mt.firmware_nose(Q)
    out = Path(a.out) if a.out else src.with_name(src.stem + "_slow.csv")
    with open(out, "w") as fh:                        # 1 micrometre: slow points are only mm apart
        fh.write("t,x,y,z,yaw_rad\n")
        for i, ((x, y), h) in enumerate(zip(Q, nose)):
            fh.write(f"{i * pc.DT:.3f},{x:.6f},{y:.6f},0.000000,{h:.4f}\n")
    r_old = np.degrees(np.abs(np.diff(np.unwrap(mt.firmware_nose(P))))) / pc.DT
    Qf = np.loadtxt(out, delimiter=",", skiprows=1)[:, 1:3]            # check what was WRITTEN
    r_new = np.convolve(np.degrees(np.abs(np.diff(np.unwrap(mt.firmware_nose(Qf))))) / pc.DT, np.ones(6) / 6, mode="same")
    print(f"{out}: {len(P) * pc.DT:.0f} s -> {len(Q) * pc.DT:.0f} s; nose turns at most ~{r_new.max():.0f} deg/s "
          f"(was up to {r_old.max():.0f} deg/s at corners)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
