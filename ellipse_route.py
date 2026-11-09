#!/usr/bin/env python3
"""
ellipse_route.py - ellipse trajectory generator for the CDE1302 mini project (v2, lab format).

Writes a CSV with columns  t,x,y,z  (seconds, metres) in the NED frame, in the same
style as trajectory/ellipse_min_snap_traj.csv from the Lab 5 repo:
    x = North, y = East, z = Down.
The lab's send_trajectory.py subtracts the first row, so the path is flown RELATIVE to
the takeoff point, and z = 0 means "hold the takeoff altitude" (0.5 m in the lab).

Geometry matches the Lab 5 ellipse by default:
  * starts at the tip of the major axis (the takeoff point),
  * major axis (2*a) along North/South, minor axis (2*b) along East/West,
  * the loop lies BEHIND the drone's nose (nose = +North), clockwise seen from above.

Speed profile: a cosine ramp up and down, so velocity starts and ends at ZERO
(no jerk at the start, like a min-snap trajectory), with constant cruise speed in between.

Example (Lab 5 size, slow):
  python ellipse_route.py --a 1.0 --b 0.5 --speed 0.15 --ramp 3 \
      --bounds -2.5 0.5 -1.0 1.0 --out ellipse_ours.csv
"""
import argparse
import csv
import math
import sys

import numpy as np


def ellipse_curve(a, b, ahead=False, ccw=False, n=40000):
    """Dense curve starting at the takeoff point (0,0), tip of the major axis.
    x = North (a along x), y = East (b along y)."""
    th = np.linspace(0.0, 2.0 * math.pi, n)
    if ccw:
        th = -th
    sign = 1.0 if ahead else -1.0          # -1 -> loop is behind the nose (Lab 5 layout)
    x = sign * a * (1.0 - np.cos(th))
    y = b * np.sin(th)
    return x, y


def arclength(x, y):
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return s


def s_of_t(t, L, v, r):
    """Distance travelled along the path vs time: cosine ramp-up (r s), cruise, ramp-down."""
    T = L / v + r
    s = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti <= r:
            s[i] = 0.5 * v * (ti - r / math.pi * math.sin(math.pi * ti / r))
        elif ti < T - r:
            s[i] = 0.5 * v * r + v * (ti - r)
        else:
            tau = max(T - ti, 0.0)
            s[i] = L - 0.5 * v * (tau - r / math.pi * math.sin(math.pi * tau / r))
    return s, T


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=float, default=1.0, help="semi-axis along North/South [m] (Lab 5: 1.0)")
    p.add_argument("--b", type=float, default=0.5, help="semi-axis along East/West [m] (Lab 5: 0.5)")
    p.add_argument("--speed", type=float, default=0.15, help="cruise speed along the path [m/s] (Lab 5 avg ~0.13)")
    p.add_argument("--ramp", type=float, default=3.0, help="time to ramp speed 0->cruise and back [s]")
    p.add_argument("--dt", type=float, default=0.05, help="time between CSV rows [s] (0.05 = the uploader's own rate)")
    p.add_argument("--ahead", action="store_true", help="put the loop AHEAD of the drone (+North) instead of behind")
    p.add_argument("--ccw", action="store_true", help="counter-clockwise instead of clockwise")
    p.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="allowed area relative to takeoff [m]; file is NOT written if the path leaves it")
    p.add_argument("--max-speed", type=float, default=0.30,
                   help="refuse speeds above this [m/s]; 0.30 = the same cap send_trajectory.py enforces")
    p.add_argument("--out", default="ellipse_ours.csv")
    p.add_argument("--preview", default="ellipse_ours_preview.png")
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    if min(args.a, args.b, args.speed, args.ramp, args.dt) <= 0:
        sys.exit("ERROR: a, b, speed, ramp and dt must be > 0")
    if args.speed > args.max_speed:
        sys.exit(f"ERROR: speed {args.speed} > safety cap {args.max_speed} m/s (raise --max-speed on purpose)")

    xd, yd = ellipse_curve(args.a, args.b, ahead=args.ahead, ccw=args.ccw)
    sd = arclength(xd, yd)
    L = sd[-1]
    if args.ramp > L / args.speed:
        sys.exit("ERROR: ramp is longer than the cruise time; reduce --ramp")

    n_rows = int(math.floor((L / args.speed + args.ramp) / args.dt))
    t = np.arange(0.0, n_rows + 1) * args.dt
    s, T = s_of_t(t, L, args.speed, args.ramp)
    if t[-1] < T - 1e-9:                     # make sure the final row is exactly the end point
        t = np.append(t, T)
        s = np.append(s, L)
    x = np.interp(s, sd, xd)
    y = np.interp(s, sd, yd)
    z = np.zeros_like(x)
    x[-1], y[-1] = 0.0, 0.0                  # close the loop exactly on the takeoff point
    x = x + 0.0                              # turns -0.0 into 0.0 so the CSV looks clean
    y = y + 0.0

    # speeds / accelerations for the safety report
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(vx, vy)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    acc = np.hypot(ax, ay)

    # ---- safety checks --------------------------------------------------------
    warnings = []
    if args.bounds:
        xmin, xmax, ymin, ymax = args.bounds
        if x.min() < xmin or x.max() > xmax or y.min() < ymin or y.max() > ymax:
            sys.exit(f"ERROR: path x[{x.min():.2f},{x.max():.2f}] y[{y.min():.2f},{y.max():.2f}] leaves "
                     f"bounds x[{xmin},{xmax}] y[{ymin},{ymax}]. Nothing written.")
    else:
        warnings.append("no --bounds given: path NOT checked against the flight area")
    if abs(x[0]) > 1e-9 or abs(y[0]) > 1e-9:
        warnings.append("first row is not (0,0)")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])
        for ti, xi, yi, zi in zip(t, x, y, z):
            w.writerow([f"{ti:.3f}", f"{round(xi, 4) + 0.0:.4f}", f"{round(yi, 4) + 0.0:.4f}", f"{zi:.4f}"])

    print(f"Rows            : {len(t)} (dt = {args.dt} s)")
    print(f"Path length     : {L:.2f} m   Duration: {t[-1]:.1f} s   Cruise: {args.speed} m/s")
    print(f"Extent N(x)     : [{x.min():.2f}, {x.max():.2f}] m   E(y): [{y.min():.2f}, {y.max():.2f}] m")
    print(f"Peak speed      : {speed.max():.3f} m/s   Peak accel: {acc.max():.3f} m/s^2")
    print(f"Start -> end    : ({x[0]:.2f},{y[0]:.2f}) -> ({x[-1]:.2f},{y[-1]:.2f})   z = 0 (holds takeoff altitude)")
    print(f"Direction       : {'counter-' if args.ccw else ''}clockwise from above; loop is "
          f"{'ahead of' if args.ahead else 'behind'} the nose (nose = +North)")
    for msg in warnings:
        print("WARNING:", msg)
    print(f"Wrote {args.out}")

    if not args.no_preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 7))
        ax.plot(y, x, "-", lw=2.5, color="#1f6aa5")
        k = len(t) // 8
        for idx in (k, 3 * k, 5 * k, 7 * k):
            ax.annotate("", xy=(y[idx + 1], x[idx + 1]), xytext=(y[idx], x[idx]),
                        arrowprops=dict(arrowstyle="-|>", color="#1f6aa5", lw=2, mutation_scale=18))
        ax.plot(0, 0, "o", ms=12, color="#c0392b")
        ax.annotate("takeoff / start / finish\n(nose points +North)", (0, 0), textcoords="offset points",
                    xytext=(12, 14 if args.ahead else 14), fontsize=9, color="#c0392b")
        if args.bounds:
            xmin, xmax, ymin, ymax = args.bounds
            ax.plot([ymin, ymax, ymax, ymin, ymin], [xmin, xmin, xmax, xmax, xmin], "r--", lw=1, label="bounds")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("East / y (m)")
        ax.set_ylabel("North / x (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Ellipse {2*args.a:.1f} m x {2*args.b:.1f} m, {t[-1]:.0f} s, v={args.speed} m/s")
        fig.tight_layout()
        fig.savefig(args.preview, dpi=150)
        print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
