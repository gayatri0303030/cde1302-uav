"""Write the measured start event that tag_mapper.py needs, from a recording made while the drone
was still sitting on its start mark (start tag_mapper recording BEFORE take-off).

  python make_start_event.py mapping_runs\\session01 --height 0.03 --sigma 0.01

--height: body origin (flight controller / IMU centre) above the floor while parked, in metres,
          measured with a ruler.  --sigma: how sure you are of that (m).
Takes the FIRST position + attitude sample of the run (the parked pose) and writes
<run>_start_event.json next to the run folder (the run itself is never modified).
Pass it to tag_mapper.py with --start-event.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def first_row(path: Path):
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("fc_time_ns"):
                return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--height", type=float, required=True, help="body origin above floor while parked (m)")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--note", default="drone parked on the taped start mark; height by ruler")
    a = ap.parse_args()

    pos = first_row(a.run / "csv" / "position.csv")
    att = first_row(a.run / "csv" / "attitude.csv")
    if pos is None or att is None:
        raise SystemExit("no position/attitude rows: was MAVLink (UDP 14550) received during the recording?")
    ev = {
        "run_id": pos["run_id"],
        "segment_id": int(pos["segment_id"]),
        "fc_time_ns": int(pos["fc_time_ns"]),
        "host_receive_ns": int(pos["host_receive_ns"]),
        "native_position_m": [float(pos["x"]), float(pos["y"]), float(pos["z"])],
        "quaternion_xyzw": [float(att["qx"]), float(att["qy"]), float(att["qz"]), float(att["qw"])],
        "initial_body_height_above_floor_m": a.height,
        "initial_body_height_sigma_m": a.sigma,
        "coordinate_convention": "NED",
        "datum_source": a.note,
    }
    out = a.run.parent / f"{a.run.name}_start_event.json"
    out.write_text(json.dumps(ev, indent=2))
    print(f"wrote {out}")
    print(f"  parked pose: x {ev['native_position_m'][0]:+.3f} y {ev['native_position_m'][1]:+.3f} "
          f"z {ev['native_position_m'][2]:+.3f}, height {a.height} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
