"""Flight safety guard: land when PX4's position estimate jumps or the drone wanders off.

Why (30 Oct, F3): 5 s into take-off PX4's position estimate jumped 2.4 m in one second (an
estimator reset, the drone itself had not moved).  The take-off controller then flew the drone
~1.5 m back-left to "return" to its arming spot, into the wall.  In 12 normal flights the largest
change between two position updates was 0.65 m and take-off drift at most 0.62 m.

Checks, on every NEW PX4 position sample (they reach the laptop about once a second):
  * jump      : moved more than max_jump_m since the previous sample AND faster than
                max_jump_speed_mps (normal flights: <= 0.65 m/s)
  * take-off  : more than takeoff_max_drift_m from the start spot before the trajectory starts
  * path      : more than max_path_error_m from where the trajectory should be, on
                path_error_samples samples in a row (0 = check off)
Any trip raises GuardTrip; run_baseline.py then sends LAND (the same path as Ctrl+C).
Pure logic, no sockets, so it can be replayed on old logs and unit tested.
"""
from __future__ import annotations

import math


class GuardTrip(RuntimeError):
    """The guard wants the drone on the ground now."""


class FlightGuard:
    def __init__(self, origin_xy, max_jump_m: float = 1.0, max_jump_speed_mps: float = 1.2,
                 takeoff_max_drift_m: float = 1.0, max_path_error_m: float = 1.2,
                 path_error_samples: int = 2, enabled: bool = True):
        self.origin = tuple(origin_xy)
        self.max_jump_m = max_jump_m
        self.max_jump_speed = max_jump_speed_mps
        self.takeoff_max_drift_m = takeoff_max_drift_m
        self.max_path_error_m = max_path_error_m
        self.path_error_samples = max(1, int(path_error_samples))
        self.enabled = enabled
        self.phase = "takeoff"          # takeoff | traj | off
        self.expected = None            # callable(t_mono) -> (x, y) local NED, during "traj"
        self._prev = None               # (x, y, key, t_mono, boot_ms)
        self._bad_path = 0
        self.worst = dict(jump_speed=0.0, takeoff_drift=0.0, path_error=0.0)

    def start_trajectory(self, expected) -> None:
        self.phase, self.expected, self._bad_path = "traj", expected, 0

    def stop(self) -> None:
        self.phase = "off"

    def update(self, x: float, y: float, t_mono: float, boot_ms=None) -> None:
        """Feed the latest telemetry; raises GuardTrip.  Repeated samples are ignored."""
        if not self.enabled or self.phase == "off" or x is None or y is None:
            return
        key = boot_ms if boot_ms is not None else (round(x, 4), round(y, 4))
        if self._prev is not None and key == self._prev[2]:
            return                                              # not a new position sample
        prev, self._prev = self._prev, (x, y, key, t_mono, boot_ms)
        if prev is not None:
            d = math.hypot(x - prev[0], y - prev[1])
            if boot_ms is not None and prev[4] is not None and boot_ms > prev[4]:
                dt = (boot_ms - prev[4]) / 1000.0
            else:
                dt = t_mono - prev[3]
            if dt > 0:
                speed = d / dt
                self.worst["jump_speed"] = max(self.worst["jump_speed"], speed)
                if d > self.max_jump_m and speed > self.max_jump_speed:
                    raise GuardTrip(f"PX4 position jumped {d:.2f} m in {dt:.1f} s ({speed:.1f} m/s): "
                                    f"estimator reset? landing")
        if self.phase == "takeoff":
            drift = math.hypot(x - self.origin[0], y - self.origin[1])
            self.worst["takeoff_drift"] = max(self.worst["takeoff_drift"], drift)
            if drift > self.takeoff_max_drift_m:
                raise GuardTrip(f"drifted {drift:.2f} m from the start spot during take-off "
                                f"(limit {self.takeoff_max_drift_m:.2f} m): landing")
        elif self.phase == "traj" and self.expected is not None and self.max_path_error_m > 0:
            ex, ey = self.expected(t_mono)
            err = math.hypot(x - ex, y - ey)
            self.worst["path_error"] = max(self.worst["path_error"], err)
            self._bad_path = self._bad_path + 1 if err > self.max_path_error_m else 0
            if self._bad_path >= self.path_error_samples:
                raise GuardTrip(f"{err:.2f} m off the planned path for {self._bad_path} updates "
                                f"(limit {self.max_path_error_m:.2f} m): landing")

    def summary(self) -> str:
        w = self.worst
        return (f"guard: biggest jump {w['jump_speed']:.2f} m/s, take-off drift {w['takeoff_drift']:.2f} m, "
                f"path error {w['path_error']:.2f} m")
