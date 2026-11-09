"""Trajectory experiment results: one row per flight, one group per trajectory.

  python compare_trajectories.py                  every run labelled E1.., E2.. (--label E1a etc.) in logs\\
  python compare_trajectories.py logs\\E1a_* logs\\E3b_*      only these runs

Measurements (written by run_baseline.py into each run's summary.json -> "metrics"):
  mission time   take-off command -> land command (s); the trajectory part is shown too
  tags detected  confirmed target tags out of the ones in --refs (tag 20 never counts)
  deviation      flown path vs the given CSV path (m): mean and max.  Uses PX4's position with
                 the tag-20 drift removed when the home tag was seen back home, else PX4 alone
  map error      mean error of the non-anchor tags after the optimisation (if it ran), for reference

Before comparing, it checks the controlled variables really were the same in every run:
camera settings and mount (config_used.yaml, camera_settings.json), box of the CSV path (width and
length), height, --set/--sensor/--pitch.  It also checks the yaw: a *_yaw path must be flown with
--yaw-follow AND the nose must really have turned (old firmware ignores the flag), a constant-yaw
path without it.  Anything that is off is printed as a WARNING.

Writes logs\\trajectory_experiment.csv (one row per flight) and logs\\trajectory_experiment.png.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:                      # the check of config_used.yaml is skipped without it
    yaml = None


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def csv_box(path: Path):
    """Extent (x min/max, y min/max) and whether the yaw column is constant."""
    try:
        d = np.genfromtxt(path, delimiter=",", comments="#", names=True)
    except (OSError, ValueError):
        return None
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    yaw_const = True
    if "yaw_rad" in d.dtype.names:
        yv = np.asarray(d["yaw_rad"])
        yaw_const = bool(np.nanmax(yv) - np.nanmin(yv) < 1e-3)
    return (round(float(x.min()), 2), round(float(x.max()), 2),
            round(float(y.min()), 2), round(float(y.max()), 2)), yaw_const


def load(run: Path, base: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    a = s.get("args", {})
    m = s.get("metrics") or {}
    traj = a.get("trajectory", "") or ""
    r = dict(run=run.name, label=a.get("label", ""), trajectory=Path(traj.replace("\\", "/")).stem,
             traj_path=traj, mission_s=fnum(m.get("mission_time_s")), traj_s=fnum(m.get("trajectory_time_s")),
             planned_s=fnum(m.get("planned_time_s")), planned_len=fnum(m.get("planned_length_m")),
             tags=m.get("tags_detected"), tags_total=m.get("tags_total"), tag_ids=m.get("tags_detected_ids", []),
             dev_mean=fnum(m.get("deviation_mean_m")), dev_max=fnum(m.get("deviation_max_m")),
             dev_px4_mean=fnum(m.get("deviation_px4_mean_m")), dev_src=m.get("deviation_source", ""),
             completed=m.get("completed"), map_after=None, yaw_follow=a.get("yaw_follow") == "True",
             swept=fnum(m.get("heading_swept_deg")),
             height=a.get("height"), sets=a.get("set"), sensor=a.get("sensor"), pitch=a.get("pitch"),
             cam=None, mount=None, sensor_read=None, box=None, yaw_const=None, note="")
    if not m:
        r["note"] = "no metrics (flown before the experiment update?)"
    comp = run / "optimization" / "comparison.csv"
    home = (s.get("home_tag") or {}).get("id")
    if comp.exists():
        with open(comp, newline="") as fh:
            errs = [fnum(x.get("after_error_m")) for x in csv.DictReader(fh)
                    if str(x.get("anchor")).lower() != "true" and str(x.get("tag_id")) != str(home)]
        errs = [e for e in errs if e is not None]
        r["map_after"] = sum(errs) / len(errs) if errs else None
    if yaml is not None and (run / "config_used.yaml").exists():
        c = yaml.safe_load((run / "config_used.yaml").read_text()) or {}
        cam = dict(c.get("camera") or {})
        r["mount"] = cam.pop("mount", None)
        r["cam"] = cam
    if (run / "camera_settings.json").exists():
        try:
            cs = json.loads((run / "camera_settings.json").read_text())
            # the chip settings the drone reported (live exposure/gain change with the light: not compared)
            r["sensor_read"] = ((cs.get("reply") or {}).get("params")) or None
        except (ValueError, AttributeError):
            pass
    for cand in (Path(traj), base / traj, base / traj.replace("\\", "/")):
        if cand.exists():
            got = csv_box(cand)
            if got:
                r["box"], r["yaw_const"] = got
            break
    return r


def check_controls(rows) -> list[str]:
    warn = []

    def same(key, what):
        vals = {json.dumps(r[key], sort_keys=True, default=str) for r in rows if r[key] is not None}
        if len(vals) > 1:
            warn.append(f"{what} differ between runs: " + "; ".join(
                f"{r['label'] or r['run']}={json.dumps(r[key], sort_keys=True, default=str)[:80]}" for r in rows))
    same("cam", "camera settings (config_used.yaml camera:)")
    same("mount", "camera mount")
    same("sensor_read", "camera sensor settings read back from the drone")
    same("sets", "--set overrides")
    same("sensor", "--sensor changes")
    same("pitch", "--pitch")
    same("height", "--height")
    same("box", "path box (x/y range of the CSV)")
    for r in rows:
        who = r["label"] or r["run"]
        yaw_path = r["yaw_const"] is False or r["trajectory"].endswith("_yaw")
        if yaw_path and not r["yaw_follow"]:
            warn.append(f"{who}: {r['trajectory']} is a yaw path but was flown WITHOUT --yaw-follow")
        if not yaw_path and r["yaw_follow"]:
            warn.append(f"{who}: {r['trajectory']} is a constant-yaw path but was flown WITH --yaw-follow")
        if r["yaw_follow"] and r["swept"] is not None and r["swept"] < 90:
            warn.append(f"{who}: --yaw-follow but the nose only turned {r['swept']:.0f} deg - the drone "
                        f"probably lacks the yaw-follow firmware, so this was really a constant-yaw flight")
        if r["completed"] is False:
            warn.append(f"{r['label'] or r['run']}: did not land normally (guard/abort) - mission time missing")
    return warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (wildcards ok); default: logs\\E<n>* runs")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--prefix", default="E", help="label letter of the experiment runs (default E)")
    ap.add_argument("--out", default=None, help="output folder (default: --logs)")
    a = ap.parse_args()
    base = Path(__file__).resolve().parent
    paths = []
    for pat in a.runs or [str(Path(a.logs) / f"{a.prefix}*")]:
        paths += [Path(p) for p in sorted(glob.glob(pat))]
    runs = [p for p in dict.fromkeys(paths) if p.is_dir() and (p / "summary.json").exists()]
    if not a.runs:
        runs = [p for p in runs if re.match(rf"{re.escape(a.prefix)}\d+[A-Za-z]*_", p.name)]
    if not runs:
        raise SystemExit(f"no runs found (label the flights --label {a.prefix}1a, {a.prefix}2a, ...)")

    def order(p):
        mm = re.match(r"[A-Za-z]+(\d+)([A-Za-z]*)_", p.name)
        return (int(mm.group(1)), mm.group(2), p.name) if mm else (999, "", p.name)
    rows = [load(p, base) for p in sorted(runs, key=order)]

    def f(x, w=6, d=2):
        return f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-"

    print("\nTrajectory experiment, one row per flight")
    head = (f"{'flight':7s} {'trajectory':28s} {'mission':>8s} {'traj':>6s} {'plan':>6s} {'length':>7s} "
            f"{'tags':>5s} {'ids':10s} {'dev mean':>8s} {'dev max':>8s} {'map err':>8s} {'nose':>5s}")
    print(head + "\n" + "-" * len(head))
    for r in rows:
        tg = f"{r['tags']}/{r['tags_total']}" if r["tags"] is not None else "-"
        print(f"{(r['label'] or r['run'])[:7]:7s} {r['trajectory'][:28]:28s} {f(r['mission_s'], 7, 1)}s "
              f"{f(r['traj_s'], 5, 1)}s {f(r['planned_s'], 5, 1)}s {f(r['planned_len'], 6, 2)}m {tg:>5s} "
              f"{' '.join(map(str, r['tag_ids']))[:10]:10s} {f(r['dev_mean'], 7, 3)}m {f(r['dev_max'], 7, 3)}m "
              f"{f(r['map_after'], 7, 3)}m {f(r['swept'], 4, 0)}d  {r['note']}")
    def shape_of(name):
        return re.sub(r"_yaw$", "", re.sub(r"^[A-Za-z]+\d+_", "", name))
    groups: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: (shape_of(r["trajectory"]), r["trajectory"].endswith("_yaw"))):
        groups.setdefault(r["trajectory"], []).append(r)        # pairs side by side: shape, then + yaw

    def mean(vals):
        v = [x for x in vals if x is not None]
        return (sum(v) / len(v), min(v), max(v), len(v)) if v else (None, None, None, 0)

    print("\nPer trajectory (mean of the repeats, [min..max])")
    summary = []
    for name, rs in groups.items():
        mt, dv, tg = mean([r["mission_s"] for r in rs]), mean([r["dev_mean"] for r in rs]), \
            mean([r["tags"] for r in rs])
        summary.append((name, rs, mt, tg, dv))
        rng = (lambda t, d: f"[{t[1]:.{d}f}..{t[2]:.{d}f}]" if t[3] > 1 else "")
        print(f"  {name:28s} n={len(rs)}  mission {f(mt[0], 6, 1)} s {rng(mt, 1):14s} tags {f(tg[0], 4, 1)} "
              f"{rng(tg, 0):8s} deviation {f(dv[0], 6, 3)} m {rng(dv, 3)}")
    warn = check_controls(rows)
    print("\nControlled variables: " + ("all the same in every run" if not warn else f"{len(warn)} WARNING(S)"))
    for w in warn:
        print("  WARNING: " + w)
    if any(r["dev_src"] and "PX4 only" in r["dev_src"] for r in rows):
        print("  note: some runs have no tag-20 drift fix, so their deviation is PX4's own view "
              "(usually smaller than the real deviation)")

    out = Path(a.out or a.logs)
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "trajectory_experiment.csv"
    with open(cp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flight", "run", "trajectory", "mission_time_s", "trajectory_time_s", "planned_time_s",
                    "planned_length_m", "tags_detected", "tags_total", "tag_ids", "deviation_mean_m",
                    "deviation_max_m", "deviation_px4_mean_m", "deviation_source", "map_error_after_opt_m",
                    "yaw_follow", "nose_turned_deg", "completed", "note"])
        for r in rows:
            w.writerow([r["label"], r["run"], r["trajectory"], r["mission_s"], r["traj_s"], r["planned_s"],
                        r["planned_len"], r["tags"], r["tags_total"], " ".join(map(str, r["tag_ids"])),
                        r["dev_mean"], r["dev_max"], r["dev_px4_mean"], r["dev_src"],
                        None if r["map_after"] is None else round(r["map_after"], 4), r["yaw_follow"], r["swept"],
                        r["completed"], r["note"]])
    print(f"\nCSV: {cp}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return 0
    names = [s[0] for s in summary]
    short = [n.split("_", 1)[0] + "\n" + shape_of(n).replace("_", "\n", 1).replace("_", " ")
             + ("\n+ yaw" if n.endswith("_yaw") else "") for n in names]
    colours = ["#f28e2b" if n.endswith("_yaw") else "#4e79a7" for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(max(10, 2.4 * len(names) + 4), 4.6))
    x = np.arange(len(names))
    panels = [("mission time (s)\ntake-off -> land command", "mission_s", 1),
              ("tags detected", "tags", 0),
              ("deviation from the given path (m)\nbar = mean, line = max", "dev_mean", 3)]
    for ax, (title, key, dec) in zip(axs, panels):
        means = []
        for i, (_, rs, *_r) in enumerate(summary):
            vals = [r[key] for r in rs if r[key] is not None]
            mu = sum(vals) / len(vals) if vals else 0.0
            means.append(mu)
            ax.scatter([i] * len(vals), vals, color="k", s=14, zorder=3)
            if key == "dev_mean":
                mx = [r["dev_max"] for r in rs if r["dev_max"] is not None]
                if mx:
                    ax.plot([i, i], [mu, max(mx)], "-", color="#e15759", lw=1.5, zorder=2)
        bars = ax.bar(x, means, 0.6, color=colours, alpha=.8)
        for b, mu in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mu, f"{mu:.{dec}f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, short, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    if rows[0]["tags_total"]:
        axs[1].set_ylim(0, rows[0]["tags_total"] + 0.6)
        axs[1].axhline(rows[0]["tags_total"], color="0.5", ls="--", lw=1)
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color="#4e79a7", label="yaw constant"),
                           Patch(color="#f28e2b", label="nose along the path")], fontsize=8, loc="upper left")
    fig.suptitle("Trajectory experiment: same box, speed, height and camera (dots = each flight)", fontsize=10)
    fig.tight_layout()
    png = out / "trajectory_experiment.png"
    fig.savefig(png, dpi=130)
    print(f"plot: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
