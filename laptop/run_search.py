"""Autonomous search mission: plan lawnmower lanes over YOUR area, fly them, report tags, land at start.

  python run_search.py --drone-id 5 --esp-ip 192.168.1.205 --area -0.5 2.5 -1.5 1.5 --speed 0.2

Area = xmin xmax ymin ymax in metres in the START frame (origin = takeoff point, +x = nose,
+y = right). Nothing is flown until you type the confirmation word. Everything else (logging,
tag table, landing, abort-lands) is run_baseline.py's fly mode.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import plan_search as ps
import run_baseline as rb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone-id", type=int, required=True)
    ap.add_argument("--esp-ip", required=True)
    ps.add_args(ap)
    ap.add_argument("--expect-ids", help="optional, e.g. 3,5,13")
    ap.add_argument("--refs", help="optional reference positions id,north_m,east_m (start frame)")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--trial", default=1)
    ap.add_argument("--telem-port", type=int, default=5005)
    ap.add_argument("--cmd-port", type=int, default=5006)
    ap.add_argument("--camera-port", type=int, default=5009)
    args = ap.parse_args()
    try:
        S, p, plan = ps.build(args)
    except ValueError as e:
        print("CANNOT PLAN:", e)
        return 2
    print(ps.summary(plan, p, S))
    plan_dir = Path(args.out_dir) / f"plan_{time.strftime('%Y%m%d_%H%M%S')}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    csv = plan_dir / "search.csv"
    ps.write_csv(plan, csv)
    ps.preview(plan, None, plan_dir / "search_plan.png")
    shutil.copy(args.config, plan_dir / "config_used.yaml")
    print(f"plan saved in {plan_dir} (look at search_plan.png first)")
    argv = ["run_baseline.py", "--drone-id", str(args.drone_id), "--esp-ip", args.esp_ip, "--mode", "fly",
            "--trajectory", str(csv), "--config", args.config, "--out-dir", args.out_dir,
            "--trial", str(args.trial), "--speed", f"{p.speed_mps:g}",
            "--telem-port", str(args.telem_port), "--cmd-port", str(args.cmd_port),
            "--camera-port", str(args.camera_port)]
    if args.expect_ids:
        argv += ["--expect-ids", args.expect_ids]
    if args.refs:
        argv += ["--refs", args.refs]
    sys.argv = argv
    return rb.main()


if __name__ == "__main__":
    sys.exit(main())
