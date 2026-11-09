"""Which floor cells can the camera see (and read a tag on) from a given drone pose?

The planner uses this to choose how far apart the search lanes can be, and the report uses it
to show coverage.  Same pinhole model and mount as the localisation code, so what the planner
believes is visible is what the detector pipeline can actually localise.

A floor tag centred at a cell counts as "visible" from a pose when
    * it is in front of the camera,
    * its centre is at least `edge_px + half the tag's apparent size` away from every image
      border (a tag cut by the border is not decoded),
    * the camera-to-tag range is at most `max_range_m` (the localisation gate rejects farther),
    * a floor tag, squashed by the shallow viewing angle, is still at least `min_tag_px`
      pixels tall (this is what limits the useful reach to about 0.6 m ahead).

Not modelled: motion blur, lighting, roll and pitch of the drone, and the 5 cm of altitude
wobble.  So this is an optimistic bound.  The offline simulation (sim_study.py) puts real
rendered tags through the real detector to check it.
"""

from __future__ import annotations

import math

import numpy as np

from apriltag_host import CAMERA_INTRINSICS as FIRMWARE_INTRINSICS, Intrinsics  # 9 Nov: calibration

from .geometry import CameraMount

FRAME_W, FRAME_H = 320, 240


def visible_mask(cells_n: np.ndarray, cells_e: np.ndarray, poses: np.ndarray, alt_m: float,
                 mount: CameraMount, tag_size_m: float = 0.12, max_range_m: float = 1.5,
                 edge_px: float = 8.0, min_tag_px: float = 14.0,
                 k: Intrinsics = FIRMWARE_INTRINSICS) -> np.ndarray:
    """Boolean array (len(poses), len(cells)): is cell j visible from pose i?

    poses: (P, 3) array of (x, y, yaw) in the local frame; cells: flat arrays of (north, east).
    """
    poses = np.atleast_2d(np.asarray(poses, dtype=np.float64))
    n = np.asarray(cells_n, dtype=np.float64)[None, :]
    e = np.asarray(cells_e, dtype=np.float64)[None, :]
    x, y, yaw = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    dn, de = n - x, e - y
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = c * dn + s * de                      # local -> body
    right = -s * dn + c * de
    p = math.radians(mount.pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    f = fwd - mount.offset_fwd_m
    r = right - mount.offset_right_m
    d = alt_m - mount.offset_down_m            # floor is alt_m below the body origin
    tx = r
    ty = -sp * f + cp * d
    tz = cp * f + sp * d
    ok = tz > 0.15
    tzs = np.where(ok, tz, 1.0)
    u = k.fx * tx / tzs + k.cx
    v = k.fy * ty / tzs + k.cy
    half = 0.5 * k.fx * tag_size_m / tzs       # half the apparent tag width in pixels
    m = edge_px + half
    ok &= (u >= m) & (u <= FRAME_W - m) & (v >= m) & (v <= FRAME_H - m)
    rng2 = tx * tx + ty * ty + tz * tz
    ok &= rng2 <= max_range_m ** 2
    # A tag lying on the floor is squashed along the viewing direction by sin(depression angle)
    # (depression = angle of the ray below the horizon, sin = height / range).  The detector
    # needs enough pixels across the SHORT side: simulation showed reliable reads at >= ~15 px
    # and hit-and-miss at 11-13 px, so cells whose tag would be shorter than min_tag_px are
    # not counted as covered.
    rng = np.sqrt(np.maximum(rng2, 1e-9))
    h_px = k.fy * tag_size_m * (d / rng) / tzs
    ok &= h_px >= min_tag_px
    return ok


def reach_ahead(alt_m: float, mount: CameraMount, **kw) -> tuple[float, float]:
    """(nearest, farthest) distance straight ahead at which a floor tag is visible."""
    ds = np.arange(-0.5, 3.0, 0.01)
    mask = visible_mask(ds, np.zeros_like(ds), np.array([[0.0, 0.0, 0.0]]), alt_m, mount, **kw)[0]
    if not mask.any():
        return 0.0, 0.0
    return float(ds[mask].min()), float(ds[mask].max())


def half_width_at(distance_m: float, alt_m: float, mount: CameraMount, **kw) -> float:
    """Half the visible cross-track width at `distance_m` ahead of the drone."""
    es = np.arange(0.0, 3.0, 0.01)
    mask = visible_mask(np.full_like(es, distance_m), es, np.array([[0.0, 0.0, 0.0]]),
                        alt_m, mount, **kw)[0]
    return float(es[mask].max()) if mask.any() else 0.0
