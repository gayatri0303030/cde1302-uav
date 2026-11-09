"""Synthetic camera view of AprilTags lying on the floor (for tests and the fake drone).

Given the drone pose, the camera mount and the tag positions, renders the 320x240 gray
image the ESP32 camera would see: each tag is a real tag16h5 marker (cv2.aruco), warped
through the pinhole model with the firmware's intrinsics.  Feeding this to the repo's
own detector and then to the localisation code closes the loop: a tag placed at (N, E)
must be reported at (N, E).

What this does and does not prove: it validates the detector -> pose -> frame-transform ->
time-sync chain.  It cannot tell you the REAL camera mount angle - that needs a tape
measure on the real drone.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from apriltag_host import CAMERA_INTRINSICS as FIRMWARE_INTRINSICS, Intrinsics  # 9 Nov: calibration

from .geometry import CameraMount, body_to_cam, local_to_body

FRAME_W, FRAME_H = 320, 240

_DICT = None


def _dictionary():
    global _DICT
    if _DICT is None:
        _DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    return _DICT


def project_floor_point(n: float, e: float, pose: tuple[float, float, float], alt: float,
                        mount: CameraMount, k: Intrinsics = FIRMWARE_INTRINSICS):
    """Local floor point (north, east) seen from a drone at pose=(x, y, yaw), altitude alt.

    Returns (u, v, z_cam) in AprilTag pixel convention, or None when behind the camera.
    """
    x, y, yaw = pose
    fwd, right = local_to_body(n, e, x, y, yaw)
    tx, ty, tz = body_to_cam(fwd, right, alt, mount)      # floor is `alt` below the body
    if tz <= 0.02:
        return None
    return k.fx * tx / tz + k.cx, k.fy * ty / tz + k.cy, tz


def render_frame(tags: dict[int, tuple[float, float]], pose: tuple[float, float, float],
                 alt: float, mount: CameraMount, tag_size_m: float = 0.12,
                 tag_yaw: float = 0.0, k: Intrinsics = FIRMWARE_INTRINSICS,
                 noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Gray (H, W) uint8 image of the floor with the tags (id -> (north, east)) on it."""
    rng = np.random.default_rng(seed)
    img = np.full((FRAME_H, FRAME_W), 120, np.uint8)
    cell = 16                                           # marker pixels per cell in the source image
    side_px = 6 * cell                                  # tag16h5: 4x4 data + 1-cell border = 6 cells
    pad = 3 * cell                                      # white quiet zone
    total = side_px + 2 * pad
    half = tag_size_m * total / side_px / 2.0           # world half-side of the padded square
    for tid, (n, e) in tags.items():
        marker = cv2.aruco.generateImageMarker(_dictionary(), tid, side_px)
        src = np.full((total, total), 255, np.uint8)
        src[pad:pad + side_px, pad:pad + side_px] = marker
        # source corners TL, TR, BR, BL  ->  world offsets (image up = tag +x, image right = tag +y)
        ct, st = math.cos(tag_yaw), math.sin(tag_yaw)
        corners_world = []
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            a, b = du * half, -dv * half               # a: along image-right, b: along image-up
            dn = ct * b - st * a
            de = st * b + ct * a
            corners_world.append((n + dn, e + de))
        dst = []
        for cn, ce in corners_world:
            pr = project_floor_point(cn, ce, pose, alt, mount, k)
            if pr is None:
                break
            dst.append((pr[0] - 0.5, pr[1] - 0.5))       # AprilTag corner convention -> OpenCV centre convention
        else:
            h = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [total, 0], [total, total], [0, total]]), np.float32(dst))
            cv2.warpPerspective(src, h, (FRAME_W, FRAME_H), dst=img, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)
