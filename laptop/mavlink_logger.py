"""Record PX4's own MAVLink stream (z, quaternion, velocity, body rates) on the laptop - no SD card.

The ESP32 firmware forwards every raw MAVLink byte it reads from PX4 to the laptop on UDP 14550
(the "QGC bridge").  This script listens there and writes each LOCAL_POSITION_NED, ATTITUDE_QUATERNION
and ATTITUDE message to a CSV, stamped with the laptop clock (the same clock raw_detections.csv uses).
Afterwards fill_from_mavlink.py copies the values into raw_detections.csv.

  1) quick check that the stream reaches this laptop (close QGroundControl first, it owns port 14550):
       python mavlink_logger.py --check
  2) during the flight, in a SECOND Miniforge Prompt, start this BEFORE run_baseline.py:
       python mavlink_logger.py --out logs\\mavlink_<anything>.csv
     and stop it with Ctrl+C after landing.

Needs: pip install pymavlink
Only listens by default.  --request asks PX4 (through the ESP32 port 8888) to send the three messages
at --rate Hz; use it only if --check shows a message missing or too slow.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

COLUMNS = ["t_epoch_rx", "msg", "time_boot_ms",
           "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
           "qw", "qx", "qy", "qz",
           "roll_rad", "pitch_rad", "yaw_rad",
           "roll_rate_radps", "pitch_rate_radps", "yaw_rate_radps"]
WANTED = {"LOCAL_POSITION_NED": 32, "ATTITUDE_QUATERNION": 31, "ATTITUDE": 30}


def connect(port: int):
    try:
        from pymavlink import mavutil
    except ImportError:
        raise SystemExit("pymavlink is not installed: pip install pymavlink")
    return mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}", source_system=254, dialect="common")


def request_rates(esp_ip: str, esp_port: int, rate_hz: float) -> None:
    from pymavlink import mavutil
    out = mavutil.mavlink_connection(f"udpout:{esp_ip}:{esp_port}", source_system=254, dialect="common")
    for name, msg_id in WANTED.items():
        out.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                  msg_id, int(1e6 / rate_hz), 0, 0, 0, 0, 0)
        print(f"asked PX4 for {name} at {rate_hz:g} Hz")
        time.sleep(0.05)


def row_of(m, t_rx: float) -> dict | None:
    k = m.get_type()
    r = {"t_epoch_rx": f"{t_rx:.4f}", "msg": k, "time_boot_ms": getattr(m, "time_boot_ms", "")}
    if k == "LOCAL_POSITION_NED":
        r.update(x_m=m.x, y_m=m.y, z_m=m.z, vx_mps=m.vx, vy_mps=m.vy, vz_mps=m.vz)
    elif k == "ATTITUDE_QUATERNION":
        r.update(qw=m.q1, qx=m.q2, qy=m.q3, qz=m.q4, roll_rate_radps=m.rollspeed,
                 pitch_rate_radps=m.pitchspeed, yaw_rate_radps=m.yawspeed)
    elif k == "ATTITUDE":
        r.update(roll_rad=m.roll, pitch_rad=m.pitch, yaw_rad=m.yaw, roll_rate_radps=m.rollspeed,
                 pitch_rate_radps=m.pitchspeed, yaw_rate_radps=m.yawspeed)
    else:
        return None
    return r


def check(conn, seconds: float) -> int:
    print(f"listening on UDP 14550 for {seconds:g} s ...")
    seen: Counter = Counter()
    t_end = time.time() + seconds
    while time.time() < t_end:
        m = conn.recv_match(blocking=True, timeout=0.5)
        if m is not None and m.get_type() != "BAD_DATA":
            seen[m.get_type()] += 1
    if not seen:
        print("NOTHING received.  Check: drone powered and on Wi-Fi, QGroundControl closed, Windows "
              "firewall allows Python, and that the firmware's QGC target IP is this laptop.")
        return 1
    for k, n in seen.most_common():
        mark = "  <-- needed" if k in WANTED else ""
        print(f"  {k:28s} {n / seconds:6.1f} Hz{mark}")
    missing = [k for k in WANTED if k not in seen]
    if missing:
        print("missing:", ", ".join(missing), "-> try: python mavlink_logger.py --check --request")
    else:
        print("OK: all needed messages arrive.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--out", help="CSV to write (default logs/mavlink_<time>.csv)")
    ap.add_argument("--check", action="store_true", help="listen 5 s and print which messages arrive")
    ap.add_argument("--request", action="store_true", help="ask PX4 to stream the messages at --rate")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--esp-ip", default="192.168.1.205")
    ap.add_argument("--esp-port", type=int, default=8888)
    args = ap.parse_args()

    conn = connect(args.port)
    if args.request:
        request_rates(args.esp_ip, args.esp_port, args.rate)
    if args.check:
        return check(conn, 5.0)

    out = Path(args.out or f"logs/mavlink_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    n: Counter = Counter()
    last_print = time.time()
    print(f"recording to {out}  (Ctrl+C to stop)")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        try:
            while True:
                m = conn.recv_match(type=list(WANTED), blocking=True, timeout=1.0)
                now = time.time()
                if m is not None:
                    r = row_of(m, now)
                    if r:
                        w.writerow(r)
                        n[r["msg"]] += 1
                if now - last_print >= 5.0:
                    f.flush()
                    print("  rows so far:", dict(n) if n else "NONE - is the stream arriving? run --check")
                    last_print = now
        except KeyboardInterrupt:
            pass
    print(f"saved {sum(n.values())} rows to {out}: {dict(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
