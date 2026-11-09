"""Laptop correction loop: makes the drone pull back onto the path when PX4 does not.

Drone 5 sits 0.2-0.3 m off the path it is sent (D1, D2 on 3 Nov: PX4's own track, towards its
nose and to its left) and PX4 never pulls it back.  This loop watches PX4's position against the
path the drone was sent and keeps sending an offset (CMD_TRAJ_NUDGE, needs the track-loop
firmware patch) that the drone adds to every trajectory setpoint:

    error e   = PX4 position - where the path says it should be (a moment earlier: lag_s)
    in the drone's own frame (ahead / right of the nose), because the offset it settles at turns
    with the nose:  I += ki * e * dt   (the learned offset, kept from one path piece to the next)
    offset    = -(I + kp * e), turned back into north/east with the nose heading NOW, <= max_m

So it is a PI controller around PX4.  The drone moves the offset in use at <= 0.15 m/s and lets it
slide back to 0 if the laptop stops sending for 1.5 s.  Firmware without the patch never answers
the probe; then the loop stays off and nothing is sent during the flight.

Every step is written to <run>/track_loop.csv.
"""
from __future__ import annotations

import csv
import math
import socket
import struct
import threading
import time
from pathlib import Path

PKT_CMD = 0x02            # WIFI_PKT_CMD
CMD_TRAJ_NUDGE = 0x0B     # must match main/wifi_task.h
REPLY = b"ENDG"


class TrackLoop:
    def __init__(self, esp_ip: str, port: int, link, rec, out_dir: Path, kp: float = 0.3, ki: float = 0.35,
                 max_m: float = 0.35, lag_s: float = 0.4, rate_hz: float = 5.0, max_err_m: float = 0.7):
        self.addr = (esp_ip, int(port))
        self.link, self.rec = link, rec
        self.kp, self.ki, self.max_m, self.lag_s = kp, ki, max_m, lag_s
        self.period = 1.0 / max(1.0, rate_hz)
        self.max_err_m = max_err_m
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 0))
        self.sock.setblocking(False)
        self.supported = False
        self._ref = None                      # (t0_mono, hover_xy, pts (N x 2, NED offsets), dt)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.I = [0.0, 0.0]                   # learned offset, drone frame (ahead, right), m
        self.e_b = [0.0, 0.0]                 # last error, drone frame
        self.out = (0.0, 0.0)                 # last offset sent, north / east
        self.applied = (0.0, 0.0)             # offset the drone says it uses
        self._last_boot = None
        self._last_sample_t = None
        self.n_samples = 0
        self.err_sum = 0.0
        self._csv_f = open(Path(out_dir) / "track_loop.csv", "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["t_mono", "segment_active", "err_n", "err_e", "err_ahead", "err_right",
                            "learned_ahead", "learned_right", "sent_n", "sent_e", "applied_n", "applied_e",
                            "heading_deg"])

    # ---------------------------------------------------------------- radio
    def _send(self, n: float, e: float) -> None:
        try:
            self.sock.sendto(struct.pack("<BBff", PKT_CMD, CMD_TRAJ_NUDGE, float(n), float(e)), self.addr)
        except OSError:
            pass

    def _read_replies(self) -> bool:
        got = False
        while True:
            try:
                data, _ = self.sock.recvfrom(64)
            except (BlockingIOError, OSError):
                return got
            if len(data) >= 20 and data[:4] == REPLY:
                _, _, an, ae = struct.unpack_from("<ffff", data, 4)
                self.applied = (an, ae)
                got = True

    def probe(self, timeout_s: float = 1.5) -> bool:
        """Send a zero offset; True if the drone answers (track-loop firmware on board)."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            self._send(0.0, 0.0)
            t = time.monotonic() + 0.25
            while time.monotonic() < t:
                if self._read_replies():
                    self.supported = True
                    return True
                time.sleep(0.02)
        return False

    # ---------------------------------------------------------------- path pieces
    def set_ref(self, t0_mono: float, hover_xy, pts, dt: float) -> None:
        """The piece the drone is flying now: setpoint(t) = hover_xy + pts[(t - t0) / dt] (NED)."""
        with self._lock:
            self._ref = (t0_mono, (float(hover_xy[0]), float(hover_xy[1])),
                         [(float(p[0]), float(p[1])) for p in pts], float(dt))

    def clear_ref(self) -> None:
        """Between pieces: keep the learned offset and keep sending it, but stop learning."""
        with self._lock:
            self._ref = None

    # ---------------------------------------------------------------- loop
    def start(self) -> None:
        if not self.supported or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="track-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self._csv_f.close()
        except OSError:
            pass

    def summary(self) -> str:
        if not self.supported:
            return "correction loop: off (this drone's firmware does not answer CMD_TRAJ_NUDGE)"
        mean = self.err_sum / self.n_samples if self.n_samples else float("nan")
        return (f"correction loop: {self.n_samples} PX4 positions used, mean error {mean:.3f} m; "
                f"learned offset {self.I[0]:+.2f} m ahead, {self.I[1]:+.2f} m right of the nose")

    def _update(self, pk, now: float, ref) -> None:
        boot = getattr(pk, "px4_pos_boot_ms", None)
        if boot is None or boot == self._last_boot:
            return                                       # no new PX4 position since the last step
        self._last_boot = boot
        age = (getattr(pk, "px4_pos_age_ms", None) or 0) / 1000.0
        t_s = now - age                                  # when PX4 measured it
        t0, hxy, pts, dt = ref
        i = int(round((t_s - t0 - self.lag_s) / dt))
        if i < 0:
            return
        i = min(i, len(pts) - 1)
        en = pk.ned_x - (hxy[0] + pts[i][0])
        ee = pk.ned_y - (hxy[1] + pts[i][1])
        h = pk.heading_rad
        c, s = math.cos(h), math.sin(h)
        ea, er = en * c + ee * s, -en * s + ee * c       # ahead / right of the nose
        self.e_b = [ea, er]
        step = 0.0 if self._last_sample_t is None else min(2.0, max(0.0, t_s - self._last_sample_t))
        self._last_sample_t = t_s
        if math.hypot(en, ee) <= self.max_err_m and step > 0:
            self.I[0] += self.ki * ea * step
            self.I[1] += self.ki * er * step
            m = math.hypot(*self.I)
            if m > self.max_m:
                self.I = [v * self.max_m / m for v in self.I]
        self.n_samples += 1
        self.err_sum += math.hypot(en, ee)
        self._row = (en, ee, ea, er)

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            pk = self.link.pkt
            with self._lock:
                ref = self._ref
            self._row = ("", "", "", "")
            if pk is not None and ref is not None:
                self._update(pk, now, ref)
            elif ref is None:
                self.e_b = [0.0, 0.0]                    # between pieces: no P term, keep I
            h = pk.heading_rad if pk is not None else 0.0
            ua = -(self.I[0] + self.kp * self.e_b[0])
            ur = -(self.I[1] + self.kp * self.e_b[1])
            m = math.hypot(ua, ur)
            if m > self.max_m:
                ua, ur = ua * self.max_m / m, ur * self.max_m / m
            c, s = math.cos(h), math.sin(h)
            un, ue = ua * c - ur * s, ua * s + ur * c
            self.out = (un, ue)
            self._send(un, ue)
            self._read_replies()
            try:
                self._csv.writerow([f"{now:.3f}", int(ref is not None), *[
                    (f"{v:+.3f}" if v != "" else "") for v in self._row],
                    f"{self.I[0]:+.3f}", f"{self.I[1]:+.3f}", f"{un:+.3f}", f"{ue:+.3f}",
                    f"{self.applied[0]:+.3f}", f"{self.applied[1]:+.3f}", f"{math.degrees(h):.1f}"])
            except (OSError, ValueError):
                pass
            next_t += self.period
            time.sleep(max(0.0, next_t - time.monotonic()))
