"""Background reader for PX4's MAVLink stream (forwarded by the ESP32 on UDP 14550).

Logs every LOCAL_POSITION_NED / ATTITUDE_QUATERNION / ATTITUDE message to <run>/mavlink.csv
(laptop wall-clock receive time + PX4 time_boot_ms) and keeps the recent ones in memory so the
recorder can attach the nearest PX4 sample to every tag detection.

Never fatal: if pymavlink is missing or the port is busy, it says so and the run continues
without these columns.
"""
from __future__ import annotations

import bisect
import csv
import threading
import time
from collections import deque
from pathlib import Path

COLUMNS = ["t_epoch_rx", "msg", "time_boot_ms",
           "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
           "qw", "qx", "qy", "qz",
           "roll_rad", "pitch_rad", "yaw_rad",
           "roll_rate_radps", "pitch_rate_radps", "yaw_rate_radps"]
WANTED = ("LOCAL_POSITION_NED", "ATTITUDE_QUATERNION", "ATTITUDE")
MAX_GAP_S = 0.25
HORIZON_S = 60.0


class MavlinkSource:
    def __init__(self, out_dir: Path, port: int = 14550, note=print):
        self.port = port
        self.out_path = Path(out_dir) / "mavlink.csv"
        self.note = note
        self.ok = False
        self.counts = {k: 0 for k in WANTED}
        self._lock = threading.Lock()
        self._buf = {k: deque() for k in WANTED}      # (t_rx, row dict)
        self._stop = threading.Event()
        self._th = None

    # ---- lifecycle -----------------------------------------------------------------------
    def start(self) -> "MavlinkSource":
        try:
            from pymavlink import mavutil
        except ImportError:
            self.note("MAVLink: pymavlink not installed (pip install pymavlink) - PX4 z/attitude columns stay empty")
            return self
        try:
            self._conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{self.port}", source_system=254,
                                                    dialect="common")
        except OSError as e:
            self.note(f"MAVLink: cannot listen on UDP {self.port} ({e}); close QGroundControl / "
                      f"mavlink_logger.py. Continuing without PX4 z/attitude columns")
            return self
        self._fh = open(self.out_path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._w.writeheader()
        self.ok = True
        self._th = threading.Thread(target=self._loop, daemon=True, name="mavlink")
        self._th.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        if self.ok:
            try:
                self._fh.close()
            except OSError:
                pass

    def summary(self) -> str:
        if not self.ok:
            return "MAVLink: off"
        return "MAVLink rows: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())

    # ---- reader --------------------------------------------------------------------------
    def _loop(self) -> None:
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                m = self._conn.recv_match(type=list(WANTED), blocking=True, timeout=0.5)
            except Exception:                                # malformed packet etc.
                continue
            now = time.time()
            if m is not None:
                try:
                    if m.get_srcSystem() != 1:               # PX4 only
                        continue
                except Exception:
                    pass
                r = self._row(m, now)
                if r is not None:
                    k = r["msg"]
                    with self._lock:
                        self._buf[k].append((now, r))
                        while self._buf[k] and now - self._buf[k][0][0] > HORIZON_S:
                            self._buf[k].popleft()
                        self.counts[k] += 1
                    self._w.writerow(r)
            if now - last_flush > 1.0:
                self._fh.flush()
                last_flush = now

    @staticmethod
    def _row(m, t):
        k = m.get_type()
        r = {"t_epoch_rx": f"{t:.4f}", "msg": k, "time_boot_ms": getattr(m, "time_boot_ms", "")}
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

    # ---- lookup --------------------------------------------------------------------------
    def nearest(self, kind: str, t: float):
        """(row, t_rx) of the sample of `kind` nearest to wall-clock time t, or (None, None)."""
        with self._lock:
            buf = list(self._buf[kind])
        if not buf:
            return None, None
        ts = [b[0] for b in buf]
        i = bisect.bisect_left(ts, t)
        j = min((x for x in (i - 1, i) if 0 <= x < len(ts)), key=lambda x: abs(ts[x] - t))
        if abs(ts[j] - t) > MAX_GAP_S:
            return None, None
        return buf[j][1], ts[j]
