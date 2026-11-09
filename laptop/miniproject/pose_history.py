"""Time-stamped drone pose buffer with interpolation.

Why this exists: a camera frame is already ~0.12-0.15 s old when it reaches the laptop
(the drone reports the age).  The drone moves in that time, so the tag must be
localised with the pose the drone had when the picture was TAKEN, not the latest one.
Telemetry (10 Hz) is stored here and looked up at the estimated capture time.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

from .geometry import wrap_pi


@dataclass(frozen=True)
class PoseSample:
    t: float        # laptop monotonic seconds
    x: float        # local north, m
    y: float        # local east, m
    yaw: float      # heading, rad
    z: float | None = None      # local down, m (only with the updated ESP32 firmware)
    vx: float | None = None     # local velocity, m/s (same)
    vy: float | None = None
    vz: float | None = None
    epoch: float | None = None  # wall-clock epoch seconds of this sample (latency-corrected)
    q: tuple | None = None      # PX4 attitude quaternion (w, x, y, z)   (firmware with attitude extension)
    rates: tuple | None = None  # PX4 body rates roll, pitch, yaw (rad/s)  (same)
    pos_boot_ms: int | None = None   # PX4 time_boot_ms of the position sample (same)
    att_boot_ms: int | None = None   # PX4 time_boot_ms of the attitude sample (same)
    q_from_px4: bool | None = None   # True = ATTITUDE_QUATERNION, False = built from ATTITUDE euler


class PoseHistory:
    def __init__(self, horizon_s: float = 30.0):
        self._buf: deque[PoseSample] = deque()
        self._horizon = horizon_s
        self._lock = threading.Lock()

    def add(self, t: float, x: float, y: float, yaw: float, z=None, vx=None, vy=None, vz=None,
            epoch: float | None = None, q=None, rates=None, pos_boot_ms=None, att_boot_ms=None,
            q_from_px4=None) -> None:
        """Store one sample; `t` must already be corrected for telemetry latency."""
        with self._lock:
            if self._buf and t <= self._buf[-1].t:
                return                                  # out of order / duplicate
            self._buf.append(PoseSample(t, x, y, yaw, z, vx, vy, vz, epoch, q, rates,
                                        pos_boot_ms, att_boot_ms, q_from_px4))
            while self._buf and t - self._buf[0].t > self._horizon:
                self._buf.popleft()

    def latest(self) -> PoseSample | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def nearest(self, t: float, max_gap_s: float = 1.0) -> PoseSample | None:
        """The RAW stored sample closest in time to t (no interpolation), or None if too far."""
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return None
        best = min(buf, key=lambda smp: abs(smp.t - t))
        return best if abs(best.t - t) <= max_gap_s else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def at(self, t: float, max_gap_s: float = 0.35, edge_tol_s: float = 0.10
           ) -> tuple[PoseSample, float] | None:
        """Pose at time t and the yaw rate (rad/s) around it.

        Returns None when there is no honest answer: no samples, t outside the stored
        range by more than edge_tol_s, or the two samples around t are more than
        max_gap_s apart (a telemetry dropout - interpolating across it would invent data).
        """
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return None
        times = [s.t for s in buf]
        if t < times[0]:
            if times[0] - t <= edge_tol_s and len(buf) >= 2:
                return buf[0], _yaw_rate(buf[0], buf[1])
            return None
        if t > times[-1]:
            if t - times[-1] <= edge_tol_s and len(buf) >= 2:
                return buf[-1], _yaw_rate(buf[-2], buf[-1])
            return None
        i = bisect_right(times, t)          # buf[i-1].t <= t < buf[i].t  (i < len unless t == last)
        if i >= len(buf):
            return buf[-1], _yaw_rate(buf[-2], buf[-1]) if len(buf) >= 2 else 0.0
        a, b = buf[i - 1], buf[i]
        if b.t - a.t > max_gap_s:
            return None
        k = (t - a.t) / (b.t - a.t)
        yaw = wrap_pi(a.yaw + k * wrap_pi(b.yaw - a.yaw))
        s = PoseSample(t, a.x + k * (b.x - a.x), a.y + k * (b.y - a.y), yaw)
        return s, _yaw_rate(a, b)


def _yaw_rate(a: PoseSample, b: PoseSample) -> float:
    dt = b.t - a.t
    return 0.0 if dt <= 0 else wrap_pi(b.yaw - a.yaw) / dt
