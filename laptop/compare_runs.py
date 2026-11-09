"""Compare runs: detection rate, accuracy and mission time per speed.

  python compare_runs.py logs/fly_A logs/fly_B ... --refs refs.csv
Needs run folders written by run_baseline.py / run_search.py. Prints a table and writes compare.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--refs")
    ap.add_argument("--out", default="compare.csv")
    a = ap.parse_args()
    refs = {}
    if a.refs:
        for ln in Path(a.refs).read_text().splitlines():
            p = ln.split("#")[0].replace(",", " ").split()
            if len(p) >= 3:
                refs[int(p[0])] = (float(p[1]), float(p[2]))
    rows = []
    for r in a.runs:
        run = Path(r)
        S = json.loads((run / "summary.json").read_text())
        tags = {int(k): v for k, v in S["tags"].items() if v.get("confirmed", True)}
        t_mission = ""
        for ln in (run / "events.jsonl").read_text().splitlines():
            e = json.loads(ln)
            if e.get("event") == "MISSION_SUMMARY":
                t_mission = e.get("text", "").split(" s ")[0].replace("mission time ", "")
        errs = [math.hypot(v["north_m"] - refs[i][0], v["east_m"] - refs[i][1]) for i, v in tags.items() if i in refs]
        rows.append([run.name, S["args"].get("speed", ""), len(tags), f"{len(tags)}/{len(refs)}" if refs else "",
                     f"{sum(errs)/len(errs):.3f}" if errs else "", f"{max(errs):.3f}" if errs else "", t_mission])
    hdr = ["run", "speed", "confirmed", "found/ref", "mean_err_m", "max_err_m", "mission_s"]
    print("  ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(x):>12}" for x in r))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
