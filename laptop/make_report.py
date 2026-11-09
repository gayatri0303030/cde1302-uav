"""Report figure for one run: 2D trajectory + start/landing + tag IDs/positions (+ optional PX4 .ulg).

  python make_report.py logs/fly_20261018_183152 [--refs refs.csv] [--ulg flight.ulg] [--area xmin xmax ymin ymax]

Writes <run>/report_2d.png, report_3d.png and report_tags.csv.  Frame = start frame
(+x/North = nose at takeoff, +y/East = right).  The .ulg overlay is the raw PX4 local position
(x North, y East, z down) shifted so its first airborne sample is the start point; pyulog needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_track(run: Path):
    p = run / "track.csv"
    if not p.exists():
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    return np.atleast_1d(a)


def load_ulg(path: str):
    try:
        from pyulog import ULog
    except ImportError:
        print("pyulog not installed (pip install pyulog); skipping the .ulg overlay")
        return None
    u = ULog(path)
    d = u.get_dataset("vehicle_local_position").data
    x, y, z = d["x"], d["y"], d["z"]
    moving = np.where(np.abs(z) > 0.15)[0]           # airborne = more than 15 cm up
    if len(moving) == 0:
        print("ulg: never airborne"); return None
    i0, i1 = moving[0], moving[-1]
    return x[i0:i1 + 1] - x[i0], y[i0:i1 + 1] - y[i0], -z[i0:i1 + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--refs")
    ap.add_argument("--ulg")
    ap.add_argument("--area", type=float, nargs=4)
    a = ap.parse_args()
    run = Path(a.run)
    S = json.loads((run / "summary.json").read_text())
    tags = {int(k): v for k, v in S["tags"].items() if v.get("confirmed", True)}
    refs = {}
    if a.refs:
        for ln in Path(a.refs).read_text().splitlines():
            ln = ln.split("#")[0].replace(",", " ").split()
            if len(ln) >= 3:
                refs[int(ln[0])] = (float(ln[1]), float(ln[2]))
    tr = load_track(run)
    ul = load_ulg(a.ulg) if a.ulg else None

    fig, ax = plt.subplots(figsize=(7, 7))
    if a.area:
        x0, x1, y0, y1 = a.area
        ax.plot([y0, y1, y1, y0, y0], [x0, x0, x1, x1, x0], "r-", lw=1, label="boundary")
    if tr is not None:
        ax.plot(tr["east_start_m"], tr["north_start_m"], "b-", lw=1.3, label="trajectory (telemetry)")
        ax.plot(tr["east_start_m"][-1], tr["north_start_m"][-1], "m^", ms=10, label="landing command point")
    if ul is not None:
        ax.plot(ul[1], ul[0], "c--", lw=1, label="PX4 .ulg")
    ax.plot(0, 0, "go", ms=10, label="start")
    for i, t in tags.items():
        ax.plot(t["east_m"], t["north_m"], "ks")
        txt = f"{i}"
        if i in refs:
            ax.plot(refs[i][1], refs[i][0], "rx")
            txt += f" ({np.hypot(t['north_m']-refs[i][0], t['east_m']-refs[i][1])*100:.0f} cm)"
        ax.annotate(txt, (t["east_m"], t["north_m"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("y: right of takeoff nose (m)"); ax.set_ylabel("x: along takeoff nose (m)")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(run.name)
    fig.savefig(run / "report_2d.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(7, 6)); ax3 = fig.add_subplot(projection="3d")
    if tr is not None:
        ax3.plot(tr["east_start_m"], tr["north_start_m"], np.full(len(tr), 0.5), "b-", label="trajectory (nominal 0.5 m)")
    if ul is not None:
        ax3.plot(ul[1], ul[0], ul[2], "c--", label="PX4 .ulg")
    for i, t in tags.items():
        ax3.scatter([t["east_m"]], [t["north_m"]], [0], c="k", marker="s"); ax3.text(t["east_m"], t["north_m"], 0, str(i))
    ax3.scatter([0], [0], [0], c="g", s=60, label="start")
    ax3.set_xlabel("y (m)"); ax3.set_ylabel("x (m)"); ax3.set_zlabel("height (m)"); ax3.legend(fontsize=8)
    fig.savefig(run / "report_3d.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    with open(run / "report_tags.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x_north_m", "y_east_m", "n_obs", "err_m_vs_ref"])
        for i, t in sorted(tags.items()):
            e = f"{np.hypot(t['north_m']-refs[i][0], t['east_m']-refs[i][1]):.3f}" if i in refs else ""
            w.writerow([i, t["north_m"], t["east_m"], t["n_obs"], e])
    print("wrote report_2d.png report_3d.png report_tags.csv in", run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
