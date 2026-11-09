"""Return to the home spot using a HOME AprilTag, not PX4's (drifting) position estimate.

Set-up: the home tag lies on the floor a known distance AHEAD of the landing spot.  The camera is
tilted 65 deg down (25 deg from vertical) and sees the floor from about under the nose to ~0.85 m
ahead at 0.5 m height; ~0.40 m in front of the start circle on its centre line keeps the tag in
the middle of the picture (a tag right under the drone sits on the image edge).

After the trajectory the drone hovers near home with the tag in view.  Each accepted sighting
gives the tag's position relative to the drone body (forward, right) from the camera alone, so

    drone (start frame) = home tag (start frame, tape-measured) - R(heading) * (forward, right)

and the drone is moved by  home - drone,  as a short trajectory relative to where it hovers
(the firmware plays offsets from its current hover point, so PX4's accumulated drift cancels).
It then looks again and repeats until within tolerance, and lands.

Pure maths here; run_baseline.py does the flying.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HomeFix:
    drone_x: float          # drone position in the start frame, from the home tag (m)
    drone_y: float
    move_x: float           # move needed, start frame (m)
    move_y: float
    n: int                  # sightings used
    spread_m: float         # how much the sightings disagree (m)

    @property
    def distance(self) -> float:
        return math.hypot(self.move_x, self.move_y)


def home_fix(sightings, tag_xy, home_xy, origin_yaw: float) -> HomeFix | None:
    """sightings: iterable of (forward_m, right_m, drone_yaw_rad) of the home tag.
    tag_xy / home_xy: start frame (x = ahead of the nose at start, y = right).
    origin_yaw: PX4 heading when the start frame was set."""
    xs, ys = [], []
    for fwd, right, yaw in sightings:
        if fwd is None or right is None or yaw is None:
            continue
        rel = yaw - origin_yaw
        c, s = math.cos(rel), math.sin(rel)
        ox, oy = fwd * c - right * s, fwd * s + right * c        # tag offset from drone, start frame
        xs.append(tag_xy[0] - ox)
        ys.append(tag_xy[1] - oy)
    if not xs:
        return None
    dx, dy = float(np.median(xs)), float(np.median(ys))
    spread = float(np.median(np.hypot(np.array(xs) - dx, np.array(ys) - dy)))
    return HomeFix(dx, dy, home_xy[0] - dx, home_xy[1] - dy, len(xs), spread)


def start_to_ned(x: float, y: float, origin_yaw: float) -> tuple[float, float]:
    """Start-frame vector (ahead of the nose, right) -> PX4 local NED (north, east)."""
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return x * c - y * s, x * s + y * c


def move_path(north: float, east: float, dt: float, max_speed: float = 0.15,
              min_s: float = 2.5, hold_s: float = 1.0) -> np.ndarray:
    """Smooth straight move (cosine speed profile) from (0, 0) to (north, east), then hold.
    Rows are (north, east, z=0) offsets, one per dt."""
    d = math.hypot(north, east)
    T = max(min_s, math.pi * d / (2.0 * max_speed))         # peak speed of the profile = pi*d/(2T)
    n = int(round(T / dt)) + 1
    tau = np.linspace(0.0, 1.0, n)
    k = 0.5 - 0.5 * np.cos(np.pi * tau)
    pts = np.zeros((n + int(round(hold_s / dt)), 3))
    pts[:n, 0], pts[:n, 1] = k * north, k * east
    pts[n:, 0], pts[n:, 1] = north, east
    return pts


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def tag_body_yaw(corners, mount, fx: float, fy: float, cx: float, cy: float):
    """Angle (rad) of the tag's 0->1 edge on the floor, in the drone body frame (atan2(right, fwd)).
    The tag is fixed in the room, so this angle changes by -dpsi when the drone's TRUE heading
    changes by dpsi: comparing it at take-off and at the end measures heading drift.  Corner rays
    are traced to the floor with the camera tilt; the height cancels out of the angle."""
    if not corners or len(corners) != 4:
        return None
    p = math.radians(mount.pitch_deg)
    sp, cp = math.sin(p), math.cos(p)
    pts = []
    for u, v in corners:
        x, y = (u - cx) / fx, (v - cy) / fy                  # camera ray (x right, y down, z = 1)
        fwd, right, down = cp - sp * y, x, sp + cp * y
        if down <= 0.02:
            return None
        pts.append((fwd / down, right / down))              # floor point at unit camera height
    e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    e2 = (pts[2][0] - pts[3][0], pts[2][1] - pts[3][1])    # opposite edge, same direction
    a1, a2 = math.atan2(e1[1], e1[0]), math.atan2(e2[1], e2[0])
    return a1 + wrap(a2 - a1) / 2.0


def median_angle(angles):
    a = [x for x in angles if x is not None]
    if not a:
        return None
    ref = a[0]
    return wrap(ref + float(np.median([wrap(x - ref) for x in a])))
