"""Coverage planner: turns a search rectangle into a flyable trajectory (lab CSV format).

Why this exists: the mini-project places AprilTags at random inside an area that is only
announced on the day.  The drone can only see tags on the floor in a small window ahead of
its nose (about 0.13 to 0.66 m ahead and ~1 m wide at the firmware's 0.5 m altitude, see
footprint.py) and the lab's trajectory mode holds the starting heading fixed.  So the search is
a coverage problem: sweep that window over the whole rectangle with as little flying as
possible, because mission time is graded.

Method (all numbers in config.yaml, section `search`):
  1. Flight rectangle = area shrunk by `boundary_margin_m` (open-loop flight drifts; the UAV
     must stay inside the boundary).
  2. Lanes parallel to the nose (x) or sideways (y), evenly spaced. The spacing is the LARGEST
     one for which the camera-window model still covers `target_coverage` of the cells that
     any dense lane pattern could cover.
  3. Lane ends are trimmed while coverage is kept; start/return approach included.
  4. Lane order and orientation: the cheapest of the candidates (start (0,0) -> lanes ->
     back to (0,0)) wins.
  5. Corners are rounded with a fillet, then the path is timed with a speed profile limited by
     cruise speed, along-track acceleration and lateral acceleration, and resampled at the
     ESP32 rate (0.05 s).

Coordinates: the START frame. Origin = where the drone sits at takeoff, +x = its nose,
+y = to the right of the nose (the same frame as the lab's trajectory CSV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .footprint import visible_mask
from .geometry import CameraMount


@dataclass(frozen=True)
class Area:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if not (self.xmax > self.xmin and self.ymax > self.ymin):
            raise ValueError("area needs xmax > xmin and ymax > ymin (metres, start frame)")

    @property
    def size(self) -> tuple[float, float]:
        return self.xmax - self.xmin, self.ymax - self.ymin

    def inset(self, m: float) -> "Area":
        return Area(self.xmin + m, self.xmax - m, self.ymin + m, self.ymax - m)

    def contains(self, x: float, y: float, tol: float = 1e-9) -> bool:
        return self.xmin - tol <= x <= self.xmax + tol and self.ymin - tol <= y <= self.ymax + tol


@dataclass
class SearchParams:
    speed_mps: float = 0.20
    boundary_margin_m: float = 0.30
    corner_radius_m: float = 0.25
    accel_mps2: float = 0.12
    lateral_accel_mps2: float = 0.12
    target_coverage: float = 0.97         # fraction of the achievable cells
    grid_m: float = 0.10
    min_tag_px: float = 14.0
    max_range_m: float = 1.5
    altitude_m: float = 0.5
    tag_size_m: float = 0.12
    dt_s: float = 0.05
    max_speed_cap_mps: float = 0.30
    max_points: int = 2400
    orientation: str = "auto"             # auto | x | y
    spacing_m: float | None = None        # force a lane spacing instead of optimising it
    hold_end_s: float = 1.0

    @staticmethod
    def from_dict(d: dict, **over) -> "SearchParams":
        known = {f for f in SearchParams.__dataclass_fields__}
        kw = {k: v for k, v in (d or {}).items() if k in known}
        kw.update({k: v for k, v in over.items() if v is not None})
        return SearchParams(**kw)


@dataclass
class SearchPlan:
    area: Area
    flight: Area
    orientation: str
    spacing_m: float
    waypoints: np.ndarray                 # (N, 2) corner points before rounding
    path: np.ndarray                      # (M, 2) rounded path, dense
    t: np.ndarray                         # (K,) seconds, uniform dt
    xy: np.ndarray                        # (K, 2) trajectory samples
    coverage: float                       # fraction of ALL area cells seen
    achievable: float                     # same, for a very dense lane pattern
    length_m: float
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])


# ------------------------------------------------------------------------------------------
# coverage
# ------------------------------------------------------------------------------------------
def _grid(area: Area, step: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(area.xmin + step / 2, area.xmax, step)
    ys = np.arange(area.ymin + step / 2, area.ymax, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return gx.ravel(), gy.ravel()


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return np.array(out)


def covered_mask(path_xy: np.ndarray, cells: tuple[np.ndarray, np.ndarray], p: SearchParams,
                 mount: CameraMount, sample_step: float = 0.08) -> np.ndarray:
    """Cells seen from at least one point of the path (yaw fixed at 0 = nose along +x)."""
    pts = _densify(np.asarray(path_xy, float), sample_step)
    poses = np.column_stack([pts, np.zeros(len(pts))])
    seen = np.zeros(len(cells[0]), dtype=bool)
    for i in range(0, len(poses), 200):                      # chunks keep memory small
        seen |= visible_mask(cells[0], cells[1], poses[i:i + 200], p.altitude_m, mount,
                             p.tag_size_m, p.max_range_m, 8.0, p.min_tag_px).any(axis=0)
    return seen


# ------------------------------------------------------------------------------------------
# lanes
# ------------------------------------------------------------------------------------------
def _lane_positions(lo: float, hi: float, spacing: float) -> np.ndarray:
    if hi - lo < 1e-6:
        return np.array([lo])
    k = int(math.ceil((hi - lo) / spacing)) + 1
    return np.linspace(lo, hi, max(k, 2))


def _lanes(flight: Area, orientation: str, spacing: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (end_a, end_b) lane segments, lowest lane coordinate first."""
    if orientation == "x":
        return [(np.array([flight.xmin, y]), np.array([flight.xmax, y]))
                for y in _lane_positions(flight.ymin, flight.ymax, spacing)]
    return [(np.array([x, flight.ymin]), np.array([x, flight.ymax]))
            for x in _lane_positions(flight.xmin, flight.xmax, spacing)]


def _snake(lanes, reverse: bool, near_first: bool, start: np.ndarray) -> np.ndarray:
    """Join lanes into one polyline, alternating direction. start -> ... -> start."""
    seq = list(reversed(lanes)) if reverse else list(lanes)
    pts = [start.copy()]
    cur = start.copy()
    for a, b in seq:
        first, second = (a, b) if np.linalg.norm(cur - a) <= np.linalg.norm(cur - b) else (b, a)
        pts.extend([first, second])
        cur = second
    pts.append(start.copy())
    return np.array(pts)


def _clean(poly: np.ndarray) -> np.ndarray:
    """Drop repeated points and collinear middles."""
    out = [poly[0]]
    for q in poly[1:]:
        if np.linalg.norm(q - out[-1]) > 1e-6:
            out.append(q)
    res = [out[0]]
    for i in range(1, len(out) - 1):
        a, b, c = res[-1], out[i], out[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            res.append(b)
    res.append(out[-1])
    return np.array(res)


def _length(poly: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


# ------------------------------------------------------------------------------------------
# rounding + timing
# ------------------------------------------------------------------------------------------
def round_corners(poly: np.ndarray, radius: float, step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Fillet every interior vertex. Returns (dense path, curvature at each path point)."""
    pts: list[np.ndarray] = [poly[0]]
    kap: list[float] = [0.0]
    for i in range(1, len(poly) - 1):
        a, p, b = poly[i - 1], poly[i], poly[i + 1]
        u, w = a - p, b - p
        lu, lw = np.linalg.norm(u), np.linalg.norm(w)
        u, w = u / lu, w / lw
        cosang = float(np.clip(np.dot(u, w), -1, 1))
        theta = math.acos(cosang)                       # interior angle between the two legs
        if theta > math.pi - 1e-3 or theta < 1e-3:       # straight, or a full reversal
            pts.append(p)
            kap.append(0.0)
            continue
        t = min(radius / math.tan(theta / 2), 0.5 * min(lu, lw))
        r_eff = t * math.tan(theta / 2)
        p1, p2 = p + u * t, p + w * t
        bis = (u + w) / np.linalg.norm(u + w)
        centre = p + bis * (r_eff / math.sin(theta / 2))
        a1 = math.atan2(*(p1 - centre)[::-1])
        a2 = math.atan2(*(p2 - centre)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        n = max(2, int(abs(da) * r_eff / step))
        for j in range(n + 1):
            ang = a1 + da * j / n
            pts.append(centre + r_eff * np.array([math.cos(ang), math.sin(ang)]))
            kap.append(1.0 / r_eff)
    pts.append(poly[-1])
    kap.append(0.0)
    return np.array(pts), np.array(kap)


def time_parameterise(path: np.ndarray, kappa: np.ndarray, p: SearchParams) -> tuple[np.ndarray, np.ndarray]:
    """Speed profile along the path -> (t, xy) resampled at p.dt_s. Starts and ends at rest."""
    seg0 = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(seg0)])
    # resample every 2 cm so the speed profile can peak in the middle of long straights
    sg = np.arange(0.0, s0[-1], 0.02)
    sg = np.append(sg, s0[-1])
    kappa = np.interp(sg, s0, kappa)
    path = np.column_stack([np.interp(sg, s0, path[:, 0]), np.interp(sg, s0, path[:, 1])])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = len(s)
    v_lim = np.full(n, p.speed_mps)
    with np.errstate(divide="ignore"):
        v_corner = np.where(kappa > 1e-9, np.sqrt(p.lateral_accel_mps2 / np.maximum(kappa, 1e-9)), np.inf)
    v_lim = np.minimum(v_lim, v_corner)
    v_lim[0] = v_lim[-1] = 0.0
    v2 = v_lim ** 2
    for i in range(1, n):                                   # forward: can't speed up faster than a
        v2[i] = min(v2[i], v2[i - 1] + 2 * p.accel_mps2 * seg[i - 1])
    for i in range(n - 2, -1, -1):                          # backward: must be able to stop
        v2[i] = min(v2[i], v2[i + 1] + 2 * p.accel_mps2 * seg[i])
    v = np.sqrt(v2)
    v_floor = 0.02
    t = np.zeros(n)
    for i in range(1, n):
        va, vb = max(v[i - 1], v_floor), max(v[i], v_floor)
        t[i] = t[i - 1] + 2 * seg[i - 1] / (va + vb)
    total = t[-1] + p.hold_end_s
    tt = np.arange(0.0, total + 1e-9, p.dt_s)
    sx = np.interp(np.minimum(tt, t[-1]), t, s)
    xy = np.column_stack([np.interp(sx, s, path[:, 0]), np.interp(sx, s, path[:, 1])])
    return tt, xy


# ------------------------------------------------------------------------------------------
# main entry
# ------------------------------------------------------------------------------------------
def plan_search(area: Area, p: SearchParams, mount: CameraMount | None = None,
                start: tuple[float, float] = (0.0, 0.0)) -> SearchPlan:
    mount = mount or CameraMount()
    notes: list[str] = []
    if p.speed_mps > p.max_speed_cap_mps + 1e-9:
        raise ValueError(f"speed {p.speed_mps} m/s exceeds the {p.max_speed_cap_mps} m/s cap of the trajectory sender")
    flight = area.inset(p.boundary_margin_m)
    if flight.xmax <= flight.xmin or flight.ymax <= flight.ymin:
        raise ValueError("area is smaller than twice the boundary margin")
    st = np.array(start, float)
    if not area.contains(*st):
        notes.append("WARNING: the start point (0, 0) is outside the area you gave")
    elif not flight.contains(*st):
        notes.append("start is inside the boundary margin; the first leg moves inward")
    cells = _grid(area, p.grid_m)

    # what is the best any pattern could do? (dense lanes in the better orientation)
    def cover(poly):
        return float(covered_mask(poly, cells, p, mount).mean())

    best_dense = 0.0
    for o in ("x", "y"):
        lanes = _lanes(flight, o, 0.2)
        best_dense = max(best_dense, cover(_snake(lanes, False, True, st)))
    achievable = best_dense
    need = p.target_coverage * achievable

    candidates = []
    orients = ("x", "y") if p.orientation == "auto" else (p.orientation,)
    for o in orients:
        extent = (flight.ymax - flight.ymin) if o == "x" else (flight.xmax - flight.xmin)
        grid_sp = [p.spacing_m] if p.spacing_m else [round(x, 2) for x in np.arange(max(extent, 0.2), 0.14, -0.05)]
        chosen = None
        for sp in grid_sp:
            lanes = _lanes(flight, o, sp)
            for rev in (False, True):
                poly = _clean(_snake(lanes, rev, True, st))
                c = cover(round_corners(poly, p.corner_radius_m)[0])
                if c >= need - 1e-9 or p.spacing_m:
                    cand = (_length(poly), o, sp, poly, c)
                    if chosen is None or cand[0] < chosen[0]:
                        chosen = cand
            if chosen is not None:
                break
        if chosen is None:
            notes.append(f"orientation {o}: target coverage not reachable")
            continue
        candidates.append(chosen)
    if not candidates:
        raise ValueError("no lane pattern reaches the coverage target; check the area and camera settings")
    length, orient, spacing, poly, cov = min(candidates, key=lambda c: c[0])

    path, kappa = round_corners(poly, p.corner_radius_m)
    cov_final = float(covered_mask(path, cells, p, mount).mean())
    t, xy = time_parameterise(path, kappa, p)
    if len(t) > p.max_points:
        raise ValueError(f"trajectory has {len(t)} points (> {p.max_points}, ESP32 limit): "
                         f"{t[-1]:.0f} s long. Raise --speed (cap {p.max_speed_cap_mps}), shrink the area, or widen lanes")
    # bounds check on the ROUNDED, timed path
    ex = (xy[:, 0].min() < area.xmin - 1e-6 or xy[:, 0].max() > area.xmax + 1e-6 or
          xy[:, 1].min() < area.ymin - 1e-6 or xy[:, 1].max() > area.ymax + 1e-6)
    inside = (xy[:, 0] >= flight.xmin - 1e-6) & (xy[:, 0] <= flight.xmax + 1e-6) & \
             (xy[:, 1] >= flight.ymin - 1e-6) & (xy[:, 1] <= flight.ymax + 1e-6)
    if ex:
        raise ValueError("planned path leaves the area; this is a bug, do not fly it")
    if not inside.all() and flight.contains(*st):
        notes.append("path leaves the margin band (start/return legs); still inside the boundary")
    return SearchPlan(area, flight, orient, spacing, poly, path, t, xy, cov_final, achievable,
                      _length(path), notes)


def write_csv(plan: SearchPlan, path) -> None:
    """Lab format: t,x,y,z (z = 0 holds the takeoff altitude), relative to the takeoff point."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("t,x,y,z\n")
        for tt, (x, y) in zip(plan.t, plan.xy):
            f.write(f"{tt:.3f},{round(x, 4) + 0.0:.4f},{round(y, 4) + 0.0:.4f},0.0000\n")


def peak_speed(plan: SearchPlan) -> float:
    d = np.linalg.norm(np.diff(plan.xy, axis=0), axis=1) / np.diff(plan.t)
    return float(d.max())
