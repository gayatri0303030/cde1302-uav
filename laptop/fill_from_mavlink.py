"""Fill the PX4 columns of raw_detections.csv from a mavlink_logger.py recording (no .ulg needed).

  python fill_from_mavlink.py logs\\fly_<time> logs\\mavlink_<time>.csv

Both files are stamped with the same laptop clock, so for every detection this takes the NEAREST
LOCAL_POSITION_NED sample (x, y, z, vx, vy, vz) and the NEAREST ATTITUDE_QUATERNION sample (quaternion
+ body rates) to camera_capture_timestamp.  No interpolation, no clock fitting.
If ATTITUDE_QUATERNION was not streamed, the quaternion is built from ATTITUDE roll/pitch/yaw instead
(column px4_quat_source says which).

Writes raw_detections_full.csv next to raw_detections.csv (the original is not touched):
  * px4_timestamp      = laptop time the position sample arrived
  * px4_x_m, px4_y_m   = from LOCAL_POSITION_NED (the ESP32 telemetry values are kept in esp_px4_x_m/y_m)
  * extra columns      : px4_time_boot_ms, px4_att_timestamp, px4_pos_offset_s, px4_att_offset_s,
                         px4_quat_source
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

MAX_GAP_S = 0.25   # leave cells empty if the nearest sample is further away than this


def euler_to_quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def nearest(ts: list[float], rows: list[dict], t: float):
    if not ts:
        return None, None
    i = bisect.bisect_left(ts, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))
    off = ts[best] - t
    return (rows[best], off) if abs(off) <= MAX_GAP_S else (None, off)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fly_dir", help="logs\\fly_<time> folder (or the raw_detections.csv itself)")
    ap.add_argument("mavlink_csv", help="CSV written by mavlink_logger.py")
    args = ap.parse_args()

    raw_path = Path(args.fly_dir)
    if raw_path.is_dir():
        raw_path = raw_path / "raw_detections.csv"
    with open(raw_path, newline="") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        dets = list(rd)
    with open(args.mavlink_csv, newline="") as f:
        mav = list(csv.DictReader(f))

    def series(name):
        rows = sorted((r for r in mav if r["msg"] == name), key=lambda r: float(r["t_epoch_rx"]))
        return [float(r["t_epoch_rx"]) for r in rows], rows

    pos_t, pos = series("LOCAL_POSITION_NED")
    aq_t, aq = series("ATTITUDE_QUATERNION")
    at_t, at = series("ATTITUDE")
    print(f"mavlink rows: position {len(pos)}, quaternion {len(aq)}, attitude {len(at)}")
    if not pos and not aq and not at:
        raise SystemExit("no usable MAVLink rows in that file")
    if dets and pos:
        t0, t1 = float(dets[0]["camera_capture_timestamp"]), float(dets[-1]["camera_capture_timestamp"])
        if t1 < pos_t[0] or t0 > pos_t[-1]:
            raise SystemExit("the MAVLink recording does not overlap the detections in time - wrong file?")

    extra = ["esp_px4_x_m", "esp_px4_y_m", "px4_time_boot_ms", "px4_att_timestamp",
             "px4_pos_offset_s", "px4_att_offset_s", "px4_quat_source"]
    out_cols = cols + [c for c in extra if c not in cols]
    filled = 0
    for d in dets:
        t = float(d["camera_capture_timestamp"])
        d["esp_px4_x_m"], d["esp_px4_y_m"] = d.get("px4_x_m", ""), d.get("px4_y_m", "")
        p, poff = nearest(pos_t, pos, t)
        if p:
            d.update(px4_timestamp=p["t_epoch_rx"], px4_x_m=p["x_m"], px4_y_m=p["y_m"], px4_z_m=p["z_m"],
                     px4_vx_mps=p["vx_mps"], px4_vy_mps=p["vy_mps"], px4_vz_mps=p["vz_mps"],
                     px4_time_boot_ms=p["time_boot_ms"], px4_pos_offset_s=f"{poff:+.4f}")
        q, qoff = nearest(aq_t, aq, t)
        if q:
            d.update(px4_qw=q["qw"], px4_qx=q["qx"], px4_qy=q["qy"], px4_qz=q["qz"],
                     px4_roll_rate_radps=q["roll_rate_radps"], px4_pitch_rate_radps=q["pitch_rate_radps"],
                     px4_yaw_rate_radps=q["yaw_rate_radps"], px4_att_timestamp=q["t_epoch_rx"],
                     px4_att_offset_s=f"{qoff:+.4f}", px4_quat_source="ATTITUDE_QUATERNION")
        else:
            a, aoff = nearest(at_t, at, t)
            if a:
                w, x, y, z = euler_to_quat(float(a["roll_rad"]), float(a["pitch_rad"]), float(a["yaw_rad"]))
                d.update(px4_qw=f"{w:.6f}", px4_qx=f"{x:.6f}", px4_qy=f"{y:.6f}", px4_qz=f"{z:.6f}",
                         px4_roll_rate_radps=a["roll_rate_radps"], px4_pitch_rate_radps=a["pitch_rate_radps"],
                         px4_yaw_rate_radps=a["yaw_rate_radps"], px4_att_timestamp=a["t_epoch_rx"],
                         px4_att_offset_s=f"{aoff:+.4f}", px4_quat_source="ATTITUDE(euler->quat)")
        if p and d.get("px4_qw"):
            filled += 1

    out = raw_path.with_name("raw_detections_full.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(dets)
    print(f"{filled}/{len(dets)} detections fully filled -> {out}")
    if filled < len(dets):
        print(f"  (rows left empty had no MAVLink sample within {MAX_GAP_S} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
