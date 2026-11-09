"""Print the drone's telemetry live (UDP 5005), every field the laptop can decode.

  python show_telemetry.py              (Ctrl+C to stop)
  python show_telemetry.py --csv t.csv  (also save every packet)

Fields the firmware does not send show as '-'.  Close run_baseline.py first (it owns port 5005).
"""
import argparse
import csv
import math
import socket
import time

from protocol import parse_telemetry

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5005)
ap.add_argument("--every", type=float, default=0.5, help="print interval (s)")
ap.add_argument("--csv")
a = ap.parse_args()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("", a.port))
s.settimeout(2.0)
w = None
if a.csv:
    fh = open(a.csv, "w", newline="")
f = lambda v, n=3: "-" if v is None else (f"{v:+.{n}f}" if isinstance(v, float) else str(v))
last = 0.0
n = 0
print(f"listening on UDP {a.port} ...")
while True:
    try:
        data, addr = s.recvfrom(4096)
    except socket.timeout:
        print("no telemetry for 2 s (drone on? run_baseline.py closed?)")
        continue
    except KeyboardInterrupt:
        break
    p = parse_telemetry(data)
    if p is None:
        continue
    n += 1
    d = {k: v for k, v in vars(p).items() if k != "vfh_blocked"}
    if a.csv:
        if w is None:
            w = csv.DictWriter(fh, fieldnames=["t_epoch", "bytes", "src"] + list(d))
            w.writeheader()
        w.writerow({"t_epoch": f"{time.time():.4f}", "bytes": len(data), "src": addr[0], **d})
    now = time.time()
    if now - last >= a.every:
        last = now
        q = p.q_wxyz if getattr(p, "q_wxyz", None) else None
        r = getattr(p, "body_rates", None)
        print(f"[{time.strftime('%H:%M:%S')}] from {addr[0]} {len(data)} B | "
              f"x {f(p.ned_x)} y {f(p.ned_y)} z {f(getattr(p, 'ned_z', None))} "
              f"hdg {math.degrees(p.heading_rad):6.1f} deg | "
              f"v {f(getattr(p, 'ned_vx', None))} {f(getattr(p, 'ned_vy', None))} {f(getattr(p, 'ned_vz', None))} | "
              f"q {'-' if not q else ' '.join(f'{c:+.3f}' for c in q)} | "
              f"rates {'-' if not r else ' '.join(f'{c:+.3f}' for c in r)} | "
              f"armed {f(getattr(p, 'px4_armed', None))} nav {p.nav_state}")
print(f"\n{n} packets")
if a.csv:
    fh.close()
