"""Comb lawnmower drawn by the team (29 Oct): lanes straight AHEAD and BACK, square corners,
last lane comes all the way down, then back along the bottom to the start.

Start frame: x = ahead of the nose at the start mark, y = right.  Start/land at (0,0).
  first leg  : (0,0) -> left end of the bottom line (0, Y_LEFT)
  lanes      : N_LANES vertical lanes from Y_LEFT to Y_RIGHT, alternately up (to X_TOP) and down
               (to X_BOT); the LAST lane goes down to x = 0
  return     : along the bottom (x = 0) back to (0,0)
Every corner is a full stop (square corners, like the drawing).  z = 0 (0.5 m hover; use
run_baseline.py --height for higher).  Camera heading: fixed forward, or along the path with
run_baseline.py --yaw-follow (needs the yaw-follow firmware).
"""
import argparse, math, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--lanes", type=int, default=8, help="vertical lines in the drawing (incl. the last full-length one)")
ap.add_argument("--y-left", type=float, default=-0.85)
ap.add_argument("--y-right", type=float, default=0.85)
ap.add_argument("--x-top", type=float, default=2.0, help="far end of the lanes (camera sees ~0.4-1.1 m further)")
ap.add_argument("--x-bot", type=float, default=0.3, help="near end of the inner lanes (above the return line)")
ap.add_argument("--speed", type=float, default=0.25)
ap.add_argument("--accel", type=float, default=0.15)
ap.add_argument("--out", default="comb_lawnmower.csv")
a = ap.parse_args()
DT = 0.05

ys = np.linspace(a.y_left, a.y_right, a.lanes)
way = [(0.0, 0.0), (0.0, ys[0])]
for i, y in enumerate(ys):
    up = i % 2 == 0
    last = i == len(ys) - 1
    if up:
        way.append((a.x_top, y))
    else:
        way.append((0.0 if last else a.x_bot, y))
    if not last:
        way.append((way[-1][0], ys[i + 1]))           # step right along the top / bottom
if way[-1][0] != 0.0:
    way.append((0.0, ys[-1]))
way.append((0.0, 0.0))


def leg(p0, p1):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    if L < 1e-9:
        return np.empty((0, 2))
    ta = a.speed / a.accel; da = 0.5 * a.accel * ta * ta
    if 2 * da > L:
        ta = math.sqrt(L / a.accel); vm = a.accel * ta; da = L / 2
    else:
        vm = a.speed
    T = 2 * ta + (L - 2 * da) / vm
    t = np.arange(DT, T + 1e-9, DT)
    s = np.clip(np.where(t < ta, 0.5 * a.accel * t**2, np.where(t < T - ta, da + vm * (t - ta), L - 0.5 * a.accel * (T - t)**2)), 0, L) / L
    return p0 + s[:, None] * (p1 - p0)


P = [np.array([way[0]], float)]
for p0, p1 in zip(way[:-1], way[1:]):
    P.append(leg(p0, p1))
P = np.vstack(P)
P = np.vstack([P, np.repeat(P[-1:], int(2.0 / DT), axis=0)])
d = np.diff(P, axis=0, prepend=P[:1]); yaw = np.zeros(len(P)); last = 0.0
for i, (dx, dy) in enumerate(d):
    if math.hypot(dx, dy) > 1e-4:
        last = math.atan2(dy, dx)
    yaw[i] = last
with open(a.out, "w") as f:
    f.write("t,x,y,z,yaw_rad\n")
    for i, ((x, y), h) in enumerate(zip(P, yaw)):
        f.write(f"{i * DT:.3f},{x:.4f},{y:.4f},0.0000,{h:.4f}\n")
L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
v = np.hypot(*np.diff(P, axis=0).T) / DT
print(f"{a.out}: {len(P)} pts, {len(P) * DT:.1f} s, path {L:.1f} m, vmax {v.max():.3f} m/s, "
      f"x {P[:, 0].min():.2f}..{P[:, 0].max():.2f}, y {P[:, 1].min():+.2f}..{P[:, 1].max():+.2f}, "
      f"lanes {a.lanes} ({ys[1] - ys[0]:.2f} m apart)")
if len(P) > 2400:
    print("WARNING: more than 2400 points - the drone refuses it; use fewer --lanes or a higher --speed")
