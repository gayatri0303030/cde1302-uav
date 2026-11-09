"""Camera -> body -> local-NED geometry for the CDE1302 UAV camera.

Frames used (all right-handed):

  camera : X right in the image, Y down in the image, Z out of the lens.
           This is what AprilTag's pose returns (protocol.TagDetection says the same).
  body   : forward, right, down (FRD) - origin at the flight controller.
  local  : NED. x = North, y = East, z = Down; origin where the drone took off.
           Telemetry ned_x / ned_y are in this frame when the start offset is (0, 0).

The camera is fixed to the body, pitched NOSE-DOWN by `pitch_deg` about the body's
right axis and shifted by a small offset.  Written out, the camera axes in body
coordinates (forward, right, down) are

    X_cam = (0,        1, 0       )
    Y_cam = (-sin(p),  0, cos(p)  )      image "down" points backwards-and-down
    Z_cam = ( cos(p),  0, sin(p)  )      optical axis points forwards-and-down

so a tag at camera coordinates (tx, ty, tz) sits at

    fwd   = off_f + cos(p)*tz - sin(p)*ty
    right = off_r + tx
    down  = off_d + sin(p)*tz + cos(p)*ty

A tag lying on the floor must have `down` equal to the drone's height above the
floor.  That is a free self-check of the mount angle and sign: log it, compare it
with the cruise altitude, and distrust the localisation if they disagree.

NOT modelled (the telemetry packet has no attitude): drone roll and pitch.  At the
low speeds used here these are a few degrees, i.e. a few centimetres at 0.7 m range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraMount:
    pitch_deg: float = 45.0         # nose-down is positive
    offset_fwd_m: float = 0.02      # camera ahead of the body origin
    offset_right_m: float = 0.0
    offset_down_m: float = 0.0


def cam_to_body(tx: float, ty: float, tz: float, mount: CameraMount) -> tuple[float, float, float]:
    """Camera-frame point -> body-frame (forward, right, down), metres."""
    p = math.radians(mount.pitch_deg)
    c, s = math.cos(p), math.sin(p)
    fwd = mount.offset_fwd_m + c * tz - s * ty
    right = mount.offset_right_m + tx
    down = mount.offset_down_m + s * tz + c * ty
    return fwd, right, down


def body_to_cam(fwd: float, right: float, down: float, mount: CameraMount) -> tuple[float, float, float]:
    """Inverse of cam_to_body (used by the simulator and the tests)."""
    p = math.radians(mount.pitch_deg)
    c, s = math.cos(p), math.sin(p)
    f = fwd - mount.offset_fwd_m
    r = right - mount.offset_right_m
    d = down - mount.offset_down_m
    tx = r
    ty = -s * f + c * d
    tz = c * f + s * d
    return tx, ty, tz


def body_to_local(fwd: float, right: float, x: float, y: float, yaw: float) -> tuple[float, float]:
    """Body-frame horizontal offset -> local NED (north, east).

    yaw is the drone heading in radians, 0 = local +x (north), clockwise positive
    seen from above (the telemetry `heading_rad`).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    return x + cy * fwd - sy * right, y + sy * fwd + cy * right


def local_to_body(n: float, e: float, x: float, y: float, yaw: float) -> tuple[float, float]:
    """Inverse of body_to_local: local (north, east) -> body (forward, right)."""
    dn, de = n - x, e - y
    cy, sy = math.cos(yaw), math.sin(yaw)
    return cy * dn + sy * de, -sy * dn + cy * de


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def localise(tx: float, ty: float, tz: float, x: float, y: float, yaw: float,
             mount: CameraMount) -> dict:
    """Tag position in local NED from its camera-frame translation and the drone pose.

    Returns fwd/right/down (body frame), north/east (local frame).
    """
    fwd, right, down = cam_to_body(tx, ty, tz, mount)
    north, east = body_to_local(fwd, right, x, y, yaw)
    return {"fwd": fwd, "right": right, "down": down, "north": north, "east": east}
