"""Compare the before/after pose-graph optimisation across flights (one row per run).

  python compare_optimisations.py --flights       only the runs labelled F1, F2, ... (--label F1 ...)
  python compare_optimisations.py                 every run in logs\\ that has an optimization folder
  python compare_optimisations.py logs\\F1_* logs\\F4_*      only these runs

For each run it reads <run>/optimization/comparison.csv (written automatically after landing) and
<run>/summary.json, then prints a table and writes:
  logs\\optimisation_by_trajectory.csv   one row per run
  logs\\optimisation_by_trajectory.png   before vs after mean error per run + per-tag errors

Mean errors use the target tags that were observed EXCEPT the anchor (its "after" error is ~0 by
construction, so including it would flatter the optimisation).  The home tag (--home-tag 20) is
never in the mean either: it is reported on its own as a check ("t20 check" = its error after the
optimisation).  "t20 fix" = the same tags' mean error after the tag-20 drift correction
(home_corrected_tags.csv), i.e. before / tag-20 fix / after optimisation side by side.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_run(run: Path) -> dict:
    row = dict(run=run.name, trajectory="", confirmed="", anchor="", status="", n_tags=0,
               mean_before_m=None, mean_after_m=None, improvement_pct=None, tags={}, note="",
               home_id=None, home_check_m=None, mean_home_fix_m=None, n_home_fix=0)
    home_fix = {}
    try:
        s = json.loads((run / "summary.json").read_text())
        h = s.get("home_tag") or {}
        if h.get("id") is not None:
            row["home_id"] = int(h["id"])
            home_fix = {int(k): v.get("error_m") for k, v in (h.get("drift_corrected_tags") or {}).items()}
        row["trajectory"] = Path(s.get("args", {}).get("trajectory", "") or "").stem
        row["confirmed"] = " ".join(str(k) for k, v in sorted(s.get("tags", {}).items(),
                                                             key=lambda kv: int(kv[0])) if v.get("confirmed"))
    except (OSError, ValueError):
        pass
    comp = run / "optimization" / "comparison.csv"
    if not comp.exists():
        row["note"] = "no optimisation (failed, skipped or --no-optimize)"
        return row
    with open(comp, newline="") as fh:
        rows = list(csv.DictReader(fh))
    before, after, used = [], [], []
    for r in rows:
        tid = int(r["tag_id"])
        b, a = fnum(r.get("before_error_m")), fnum(r.get("after_error_m"))
        is_anchor = str(r.get("anchor")).lower() == "true"
        row["status"] = r.get("status", row["status"])
        if is_anchor:
            row["anchor"] = tid
        if tid == row["home_id"]:                    # check only, never in the mean
            row["home_check_m"] = a
            continue
        row["tags"][tid] = (b, a, is_anchor)
        if not is_anchor and b is not None and a is not None:
            before.append(b)
            after.append(a)
            used.append(tid)
    row["n_tags"] = len(before)
    fix = [fnum(home_fix.get(t)) for t in used]
    fix = [x for x in fix if x is not None]
    if fix:
        row["mean_home_fix_m"] = sum(fix) / len(fix)
        row["n_home_fix"] = len(fix)
    if before:
        row["mean_before_m"] = sum(before) / len(before)
        row["mean_after_m"] = sum(after) / len(after)
        if row["mean_before_m"] > 0:
            row["improvement_pct"] = 100.0 * (1 - row["mean_after_m"] / row["mean_before_m"])
    if row["status"] and not row["status"].startswith("converged"):
        row["note"] = f"solver: {row['status']} (after = before)"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (wildcards ok); default: every folder in --logs")
    ap.add_argument("--flights", action="store_true",
                    help="only runs whose folder starts with F<number>_ (made with --label F1 etc.)")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default=None, help="output folder (default: --logs)")
    a = ap.parse_args()
    paths = []
    for pat in a.runs or [str(Path(a.logs) / "*")]:
        paths += [Path(p) for p in sorted(glob.glob(pat))]
    import re
    runs = [p for p in paths if p.is_dir() and (p / "summary.json").exists()
            and (a.runs or a.flights or (p / "optimization").exists())]
    if a.flights:                      # case-sensitive on purpose: Windows glob would also match fly_*
        runs = [p for p in runs if re.match(r"F\d+[A-Za-z]*_", p.name)]

    def order(p):                      # F1, F2, ..., F10, F10b in flight order, then by time
        m = re.match(r"F(\d+)([A-Za-z]*)_", p.name)
        return (0, int(m.group(1)), m.group(2), p.name) if m else (1, 0, "", p.name)
    runs = sorted(dict.fromkeys(runs), key=order)
    if not runs:
        raise SystemExit("no runs found (a run needs summary.json; without arguments only runs with an "
                         "optimization folder are listed)")
    rows = [load_run(p) for p in runs]
    all_ids = sorted({t for r in rows for t in r["tags"]})

    def f(x, w=6, d=3):
        return f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-"

    head = (f"{'run':38s} {'trajectory':26s} {'confirmed':10s} {'anchor':>6s} {'tags':>4s} {'before':>7s} "
            f"{'t20 fix':>7s} {'after':>7s} {'better':>7s} {'t20 chk':>7s}")
    print("\nOptimisation by run (mean horizontal error of the non-anchor target tags, metres;"
          "\n t20 fix = after the tag-20 drift correction, t20 chk = home tag's own error after optimisation)")
    print(head)
    print("-" * len(head))
    for r in rows:
        imp = f"{r['improvement_pct']:+6.0f}%" if r["improvement_pct"] is not None else "      -"
        print(f"{r['run'][:38]:38s} {r['trajectory'][:26]:26s} {r['confirmed']:10s} {str(r['anchor']):>6s} "
              f"{r['n_tags']:4d} {f(r['mean_before_m'], 7)} {f(r['mean_home_fix_m'], 7)} {f(r['mean_after_m'], 7)} "
              f"{imp} {f(r['home_check_m'], 7)}  {r['note']}")
    print("\nPer tag: before -> after (m), * = anchor")
    for r in rows:
        cells = []
        for t in all_ids:
            if t in r["tags"]:
                b, af, anc = r["tags"][t]
                cells.append(f"tag {t}: {f(b, 5, 2)} -> {f(af, 5, 2)}{'*' if anc else ' '}")
            else:
                cells.append(f"tag {t}:   not seen   ")
        if r["home_id"] is not None:
            cells.append(f"home tag {r['home_id']} (check): after {f(r['home_check_m'], 5, 2)}")
        print(f"  {r['run'][:38]:38s} " + "  ".join(cells))

    out = Path(a.out or a.logs)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "optimisation_by_trajectory.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "trajectory", "confirmed_tags", "anchor_tag", "tags_compared",
                    "mean_before_error_m", "mean_home_fix_error_m", "mean_after_error_m", "improvement_pct",
                    "home_tag", "home_tag_check_after_m", "solver_status", "note"]
                   + [f"tag{t}_{k}" for t in all_ids for k in ("before_m", "after_m")])
        for r in rows:
            per = []
            for t in all_ids:
                b, af, _ = r["tags"].get(t, (None, None, False))
                per += [f"{b:.4f}" if b is not None else "", f"{af:.4f}" if af is not None else ""]
            w.writerow([r["run"], r["trajectory"], r["confirmed"], r["anchor"], r["n_tags"],
                        f"{r['mean_before_m']:.4f}" if r["mean_before_m"] is not None else "",
                        f"{r['mean_home_fix_m']:.4f}" if r["mean_home_fix_m"] is not None else "",
                        f"{r['mean_after_m']:.4f}" if r["mean_after_m"] is not None else "",
                        f"{r['improvement_pct']:.1f}" if r["improvement_pct"] is not None else "",
                        "" if r["home_id"] is None else r["home_id"],
                        f"{r['home_check_m']:.4f}" if r["home_check_m"] is not None else "",
                        r["status"], r["note"]] + per)
    print(f"\nCSV: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return 0
    plotted = [r for r in rows if r["mean_before_m"] is not None]
    if not plotted:
        return 0
    def short(r):
        return f"{r['run'].split('_')[0]} {r['trajectory']}"
    labels = [f"{short(r)}\n(anchor {r['anchor']}, {r['n_tags']} tags)" for r in plotted]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(max(8, 1.3 * len(plotted) + 3), 8.5),
                                 gridspec_kw=dict(height_ratios=[1, 1]))
    x = list(range(len(plotted)))
    has_fix = any(r["mean_home_fix_m"] is not None for r in plotted)
    wd = 0.27 if has_fix else 0.38
    off = wd if has_fix else wd / 2
    b1 = ax.bar([i - off for i in x] if has_fix else [i - wd / 2 for i in x],
                [r["mean_before_m"] for r in plotted], wd, label="before optimisation", color="#f28e2b")
    groups = [b1]
    if has_fix:
        b3 = ax.bar(x, [r["mean_home_fix_m"] or 0.0 for r in plotted], wd,
                    label="tag-20 drift fix", color="#59a14f")
        groups.append(b3)
    b2 = ax.bar([i + off for i in x] if has_fix else [i + wd / 2 for i in x],
                [r["mean_after_m"] for r in plotted], wd,
                label=(f"after optimisation (tag {plotted[0]['anchor']} anchor)"
                       if len({str(r['anchor']) for r in plotted}) == 1 else "after optimisation (anchor per run)"),
                color="#4e79a7")
    groups.append(b2)
    for bars in groups:
        for bar, r in zip(bars, plotted):
            if bars is groups[1] and has_fix and r["mean_home_fix_m"] is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0, "no t20", ha="center", va="bottom", fontsize=7)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("mean error, non-anchor tags (m)")
    ax.set_title("Before vs tag-20 drift fix vs after optimisation, per run" if has_fix
                 else "Before vs after optimisation, per run")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)
    # per tag: hollow circle = before, filled square = after, joined by a line (anchor left out)
    colours = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]
    n_ids = max(1, len(all_ids))
    for k, t in enumerate(all_ids):
        c = colours[k % len(colours)]
        off = (k - (n_ids - 1) / 2) * min(0.18, 0.8 / n_ids)
        first = True
        for i, r in enumerate(plotted):
            b, af, anc = r["tags"].get(t, (None, None, False))
            if b is None or af is None or anc:
                continue
            bx.plot([i + off, i + off], [b, af], "-", color=c, alpha=.6)
            bx.plot(i + off, b, "o", mfc="none", mec=c, ms=6)
            bx.plot(i + off, af, "s", color=c, ms=6, label=f"tag {t}" if first else None)
            first = False
    bx.set_xticks(x, [short(r) for r in plotted], fontsize=8)
    bx.set_xlim(-0.6, len(plotted) - 0.4)
    bx.set_ylabel("error per tag (m)")
    bx.set_title("Per tag (anchor left out): hollow circle = before, square = after")
    bx.grid(axis="y", alpha=.3)
    bx.legend(fontsize=8, ncol=len(all_ids), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    png = out / "optimisation_by_trajectory.png"
    fig.savefig(png, dpi=130)
    print(f"plot: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
