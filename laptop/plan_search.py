"""Plan a lawnmower search over a rectangle and write the lab trajectory CSV (+ preview PNG).

Frame = START frame: (0, 0) is where the drone sits at takeoff, +x = its nose, +y = to its right.
Give the area as xmin xmax ymin ymax in metres in that frame.

  python plan_search.py --area -0.5 3.0 -1.5 1.5 --speed 0.2 --out search.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from miniproject.footprint import reach_ahead
from miniproject.planner import Area, SearchParams, plan_search, write_csv, peak_speed
from miniproject.settings import DEFAULT_CONFIG, load_settings


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--area", type=float, nargs=4, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="search rectangle in the START frame (m): +x = nose, +y = right")
    ap.add_argument("--speed", type=float, help="cruise speed m/s (default config search.speed_mps, max 0.30)")
    ap.add_argument("--margin", type=float, help="stay this far inside the boundary (m)")
    ap.add_argument("--orientation", choices=["auto", "x", "y"], help="lane direction (x = along the nose)")
    ap.add_argument("--spacing", type=float, help="force lane spacing (m)")
    ap.add_argument("--target-coverage", type=float)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))


def build(args):
    S = load_settings(args.config)
    p = SearchParams.from_dict(S.search, speed_mps=args.speed, boundary_margin_m=args.margin,
                               orientation=args.orientation, spacing_m=args.spacing,
                               target_coverage=args.target_coverage, tag_size_m=S.tag_size_m,
                               max_range_m=S.gate.max_range_m, dt_s=S.flight.dt_s)
    x0, x1, y0, y1 = args.area
    return S, p, plan_search(Area(x0, x1, y0, y1), p, S.mount)


def summary(plan, p, S) -> str:
    near, far = reach_ahead(p.altitude_m, S.mount, tag_size_m=p.tag_size_m, max_range_m=p.max_range_m, min_tag_px=p.min_tag_px)
    lines = [f"area x[{plan.area.xmin:+.2f},{plan.area.xmax:+.2f}] y[{plan.area.ymin:+.2f},{plan.area.ymax:+.2f}] m "
             f"-> flight rectangle shrunk by {p.boundary_margin_m:.2f} m",
             f"camera window (model): {near:.2f}-{far:.2f} m ahead of the nose",
             f"lanes along {plan.orientation}, spacing {plan.spacing_m:.2f} m, path {plan.length_m:.1f} m, "
             f"{plan.duration_s:.0f} s, {len(plan.t)} points, peak speed {peak_speed(plan):.3f} m/s",
             f"modelled coverage {plan.coverage*100:.0f} % of the area "
             f"(best any pattern gets: {plan.achievable*100:.0f} %)"]
    lines += ["NOTE: " + n for n in plan.notes]
    return "\n".join(lines)


def preview(plan, tags, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    a, f = plan.area, plan.flight
    ax.plot([a.ymin, a.ymax, a.ymax, a.ymin, a.ymin], [a.xmin, a.xmin, a.xmax, a.xmax, a.xmin], "r-", label="boundary")
    ax.plot([f.ymin, f.ymax, f.ymax, f.ymin, f.ymin], [f.xmin, f.xmin, f.xmax, f.xmax, f.xmin], "r:", label="flight margin")
    ax.plot(plan.xy[:, 1], plan.xy[:, 0], "b-", lw=1.2, label="planned path")
    ax.plot(0, 0, "go", ms=9, label="start / land")
    for i, (x, y) in (tags or {}).items():
        ax.plot(y, x, "ks"); ax.annotate(str(i), (y, x))
    ax.set_xlabel("east / right  y (m)"); ax.set_ylabel("north / nose  x (m)")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=7); ax.grid(alpha=.3)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--out", default="search.csv")
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    try:
        S, p, plan = build(args)
    except ValueError as e:
        print("CANNOT PLAN:", e); return 2
    print(summary(plan, p, S))
    write_csv(plan, args.out)
    print("wrote", args.out)
    if args.preview:
        png = str(Path(args.out).with_suffix(".png"))
        preview(plan, None, png); print("wrote", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
