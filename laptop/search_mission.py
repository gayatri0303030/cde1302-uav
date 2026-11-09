"""Search mission for the real test area: outward-facing ellipse, backup ellipse, accurate landing.

  python search_mission.py                 plan from config.yaml (search_area:) and draw
                                           trajectories\\search_plan.png (no flying)
  python run_baseline.py ... --mission search --height 0.8        fly it (see the run sheet)

The flight (all positions from the pad centre: x ahead, y right; config.yaml search_area:):
  1. take off, look at home tag 20 (re-zero), nose FORWARD, fly straight to the bottom tip T of the
     ellipse, turn the nose round to face OUT, fly once round the ellipse with the nose always
     pointing outward (90 deg right of travel), back to T.                      [segment 1]
  2. if fewer than `targets` (3) tags besides 20 and the anchor are confirmed: fly an ADJUSTED,
     slower ellipse through T, sized to put the spots not yet found in the middle of the camera's
     view (or to cover the most unseen ground), back to T.  Up to --search-retries times. [2]
  3. nose stays on the start: fly back towards the pad to a point where tag 20 is ~1 m ahead,
     look at it, work out where the drone really is, fly onto the pad centre and land.  [3]

Turns: the path itself turns at <= 25 deg/s (many points, slower in bends) and the nose turns with
the yaw v2 firmware limits (config.yaml yaw:).  Needs the yaw v2 firmware for the outward nose.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_trajectory as mt     # noqa: E402
import plan_coverage as pc       # noqa: E402

_YAW_RATE, _YAW_ACC = mt.yaw_limits()          # config.yaml yaw: (3 Nov: 15 deg/s, 15 deg/s^2)
TURN_DPS = min(25.0, 0.85 * math.degrees(_YAW_RATE))   # the path never turns faster than the nose can
# at T: hover while the nose swings 180 deg from forward to outward (accelerate, cruise, stop) + 0.5 s
DWELL_S = math.pi / _YAW_RATE + _YAW_RATE / _YAW_ACC + 0.5
TOP_HOLD_S = 2.0                 # hover at the far tip, nose on the anchor
ENOUGH_FRAMES = 16               # ellipse 1: every spot/anchor in view this many frames (2 s; 3 confirm it), then the fastest


# ------------------------------------------------------------------ area
def load_area(config: Path | str = HERE / "miniproject" / "config.yaml", targets: int | None = None):
    import yaml
    c = yaml.safe_load(Path(config).read_text()) or {}
    s = c.get("search_area") or {}
    an = c.get("anchors") or {}
    a = SimpleNamespace(
        x=tuple(float(v) for v in s.get("x", (0.9, 4.5))),
        y=tuple(float(v) for v in s.get("y", (-1.8, 1.35))),
        spots=[tuple(map(float, p)) for p in s.get("spots", [])],
        targets=int(targets if targets is not None else s.get("targets", 3)),
        anchor_id=int(an.get("map_tag", 5)),
        anchor_at=tuple(float(v) for v in s.get("anchor_at", an.get("map_tag_at", (4.2, 0.0)))),
        home_id=int(an.get("home_tag", 20)),
        home_at=tuple(float(v) for v in an.get("home_tag_at", (0.75, 0.0))),
        height=float(s.get("height_m", 0.8)),
        speed=float(s.get("speed_mps", 0.2)),
        search_speed=float(s.get("search_speed_mps", 0.15)),
        look_dist=float(s.get("look_distance_m", 1.0)),
        pull=float(s.get("outward_pull_m", 0.0)),
        drift=float(s.get("outward_drift_m", 0.2)))
    a.expect_drift = a.drift          # set to 0 when the correction loop is on (it removes the drift)
    # all map anchors (5 Nov: tag 5 + extra_map_tags 11, 12): the search needs >= 1 of them
    a.anchors = {a.anchor_id: a.anchor_at}
    for k, v in (an.get("extra_map_tags") or {}).items():
        a.anchors[int(k)] = tuple(float(q) for q in v)
    # ---- config.yaml tuning: (5 Nov) - the knobs to change after each run; they win over search_area:
    t = c.get("tuning") or {}

    def g(key, default):
        v = t.get(key)
        return default if v is None else v
    a.speed = float(g("speed_mps", a.speed))
    a.search_speed = float(g("search_speed_mps", a.search_speed))
    a.look_dist = float(g("look_distance_m", a.look_dist))
    a.lead_speed = float(g("lead_speed_mps", a.speed))
    a.return_speed = float(g("return_speed_mps", 0.2))
    a.tip_x = t.get("tip_x_m")
    a.tip_x_min = float(g("tip_x_min_m", 1.3))
    a.tip_x_max = float(g("tip_x_max_m", 1.6))
    a.tip_turn_s = float(g("tip_turn_s", DWELL_S))
    a.top_hold_s = float(g("top_hold_s", TOP_HOLD_S))
    a.enough_frames = int(g("enough_frames", ENOUGH_FRAMES))

    def loop(key):                                  # [length, width] in m, or null = the planner picks
        v = t.get(key)
        return None if v is None else (float(v[0]), float(v[1]))
    a.main_loop = loop("main_loop_m")               # 5 Nov (2): main 2.2 x 0.5, backup 0.6 x 0.5
    a.backup_loop = loop("backup_loop_m")
    a.backup_tip_x = t.get("backup_tip_x_m")       # null = same tip as the main loop
    a.main_shape = str(t.get("main_shape") or "ellipse")   # 5 Nov (3): "square" = rounded square
    a.main_centre = None if t.get("main_centre_m") is None else tuple(float(v) for v in t["main_centre_m"])
    a.fillet = float(g("main_fillet_m", 0.4))
    a.waist = float(g("main_waist_m", 1.3))           # dog bone: width in the middle
    a.main_nose = str(t.get("main_nose") or "out")    # 7 Nov: "in" = camera faces INTO the loop
    a.notch = float(g("main_notch_m", 0.15))          # dog bone: how far each end is dented in
    a.opt_min_frames = int(g("opt_min_frames", 30))
    a.turn_home_dps = float(g("turn_home_dps", 20.0))  # 7 Nov 19:10: the closing U-turn's nose rate
    # 7 Nov 21:30 "lanes": nose along the path, up one column of spots, across the top, down the other
    a.lanes_y = tuple(float(v) for v in (t.get("lanes_y_m") or (-1.55, 1.10)))     # [left, right] column
    a.lanes_x = tuple(float(v) for v in (t.get("lanes_x_m") or (1.37, 4.2)))       # [near end, top line]
    a.apex_x = float(g("apex_x_m", 2.2))                                           # where the V meets
    a.apex_turn_r = float(g("apex_turn_r_m", 0.12))   # 7 Nov 23:56: the turn on the spot at the apex
    return a


def pulled(prm: dict, pull: float) -> dict:
    """The ellipse to SEND so the drone FLIES prm.  With the nose pointing out of the loop drone 5
    drifts ~0.2 m the way its nose points (D1 run, 3 Nov: PX4 track 0.21 m outside the plan all the
    way round), so the commanded ellipse is shrunk by `pull` on every side (tip moved in, both
    half-axes smaller); the drone then drifts back out onto the planned one."""
    c = max(0.0, min(pull, prm["b"] - 0.12, prm["a"] - 0.30))
    return dict(xB=prm["xB"] + c, yc=prm["yc"], a=prm["a"] - c, b=prm["b"] - c)


# ------------------------------------------------------------------ paths
def rounded_square(xB: float, yc: float, a: float, b: float, r: float, step: float = 0.0025,
                   ys: float | None = None) -> np.ndarray:
    """Rounded rectangle 2a long (ahead) x 2b wide, near side through T=(xB, yc), corners filleted with
    radius r.  Same start and direction as ellipse(): from T heading +y (right), inside on the
    drone's LEFT (nose 90 deg right of travel faces OUT), once round, back to T.
    ys (7 Nov 19:10): start / end on the near side at (xB, ys) instead of (xB, yc)."""
    r = max(0.05, min(r, a - 0.01, b - 0.01))
    if ys is None or not (yc - b + r <= ys <= yc + b - r):
        ys = yc
    x0, x1, y0, y1 = xB, xB + 2 * a, yc - b, yc + b

    def line(p, q):
        n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / step))
        return np.linspace(p, q, n, endpoint=False)

    def arc(cx, cy, t0, t1):
        n = max(4, int(abs(t1 - t0) * r / step))
        t = np.linspace(t0, t1, n, endpoint=False)
        return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
    parts = [line((x0, ys), (x0, y1 - r)), arc(x0 + r, y1 - r, math.pi, math.pi / 2),
             line((x0 + r, y1), (x1 - r, y1)), arc(x1 - r, y1 - r, math.pi / 2, 0.0),
             line((x1, y1 - r), (x1, y0 + r)), arc(x1 - r, y0 + r, 0.0, -math.pi / 2),
             line((x1 - r, y0), (x0 + r, y0)), arc(x0 + r, y0 + r, -math.pi / 2, -math.pi),
             line((x0, y0 + r), (x0, ys)), np.array([[x0, ys]])]
    return np.vstack(parts)


def dog_bone(xB: float, yc: float, a: float, b: float, waist: float, notch: float = 0.15,
             step: float = 0.0025) -> np.ndarray:
    """7 Nov: dog bone, 2a long (ahead) x 2b wide at the lobes, `waist` wide in the middle, each end
    dented in by `notch`.  Bounding box: xB .. xB+2a, yc-b .. yc+b.  Starts at the near dent's middle
    (xB+notch, yc) heading +y and goes round like ellipse() (inside on the drone's LEFT, so the nose,
    90 deg right of travel, faces OUT).  Corners rounded by corner cutting (Chaikin)."""
    w = min(waist / 2, b - 0.05)
    u = [(-1.00 + notch / a, 0.0), (-1.00 + notch / a, 0.19), (-1.0, 0.69), (-0.94, 1.0), (-0.67, 0.98),
         (-0.39, w / b), (0.39, w / b), (0.67, 0.98), (0.94, 1.0), (1.0, 0.69), (1.00 - notch / a, 0.19),
         (1.00 - notch / a, -0.19), (1.0, -0.69), (0.94, -1.0), (0.67, -0.98), (0.39, -w / b), (-0.39, -w / b),
         (-0.67, -0.98), (-0.94, -1.0), (-1.0, -0.69), (-1.00 + notch / a, -0.19)]
    P = np.array(u, float)
    for _ in range(5):                                # Chaikin: cut every corner, 5 times -> smooth
        Q = np.roll(P, -1, axis=0)
        P = np.empty((2 * len(P), 2))
        P[0::2] = 0.75 * np.roll(Q, 1, axis=0) + 0.25 * Q
        P[1::2] = 0.25 * np.roll(Q, 1, axis=0) + 0.75 * Q
    P = (P - P.min(axis=0)) / (P.max(axis=0) - P.min(axis=0))      # stretch back to the full box
    P = np.c_[xB + 2 * a * P[:, 0], yc - b + 2 * b * P[:, 1]]
    k = int(np.argmin(np.hypot(P[:, 0] - (xB + notch), P[:, 1] - yc) + 10 * (P[:, 0] > xB + a)))
    P = np.vstack([P[k:], P[:k], P[k:k + 1]])        # start (and end) at the near dent
    if P[min(5, len(P) - 1), 1] < P[0, 1]:            # make sure it sets off to the right (+y)
        P = P[::-1]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    S = np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.c_[np.interp(S, s, P[:, 0]), np.interp(S, s, P[:, 1])]


def main_loop(xB, yc, a, b, area):
    """The main loop: an ellipse, or (tuning main_shape) a rounded square / dog bone of the same size."""
    shape = getattr(area, "main_shape", "ellipse")
    if shape == "square":
        nose_in = getattr(area, "main_nose", "out") == "in"
        # 7 Nov 19:10: nose in -> start / end on the near side straight ahead of the home tag (the
        # closing U-turn then ends on the look point, facing the pad)
        ys = float(area.home_at[1]) if nose_in and hasattr(area, "home_at") else None
        P = rounded_square(xB, yc, a, b, getattr(area, "fillet", 0.4), ys=ys)
        if nose_in:
            # 7 Nov: nose INWARD.  The firmware turns the nose 90 deg right of travel; going round the
            # other way puts "right of travel" on the INSIDE.  Same start / end point T.
            P = P[::-1].copy()
        return P
    if shape in ("dogbone", "dog_bone", "bone"):
        return dog_bone(xB, yc, a, b, getattr(area, "waist", 2 * b * 0.8), getattr(area, "notch", 0.15))
    return ellipse(xB, yc, a, b)


def ellipse(xB: float, yc: float, a: float, b: float) -> np.ndarray:
    """Dense ellipse from its bottom tip T=(xB, yc), first heading +y (right), inside on the drone's
    LEFT, so 'nose 90 deg right of travel' faces OUT; once round, back to T.  0.25 deg steps."""
    ph = np.radians(np.arange(0, 360.0001, 0.25))
    return np.c_[xB + a - a * np.cos(ph), yc + b * np.sin(ph)]


def _window_and_dwell(Q: np.ndarray, T, dwell_s: float):
    """Nose-outward from the point nearest T (on the way in) to the end; hover dwell_s there first."""
    k = int(np.argmin(np.hypot(*(Q - np.asarray(T)).T)[: len(Q) // 2 + 1]))
    Q2 = np.vstack([Q[:k + 1], np.repeat(Q[k:k + 1], int(round(dwell_s / pc.DT)), axis=0), Q[k + 1:]])
    return Q2, (k, len(Q2))


def _top_hold(Q: np.ndarray, window, hold_s: float = TOP_HOLD_S):
    """Hover hold_s at the far tip of the ellipse: the nose is then straight ahead, on the anchor
    (D1: the anchor got only 4 frames while the nose swept past it)."""
    if hold_s <= 0:
        return Q, window
    k = window[0] + int(np.argmax(Q[window[0]:, 0]))
    Q2 = np.vstack([Q[:k + 1], np.repeat(Q[k:k + 1], int(round(hold_s / pc.DT)), axis=0), Q[k + 1:]])
    return Q2, (window[0], window[1] + len(Q2) - len(Q))


def u_turn(area, start) -> np.ndarray | None:
    """7 Nov 19:10: after a nose-IN loop (it ends on the near side heading -y, nose +x = away from the
    pad) a half circle to the right, from the loop's end onto the look point, arriving heading +y.
    With the nose 90 deg right of travel it swings smoothly from +x round to -x: it ends ON the look
    point FACING the pad (tag 23 ahead), no detour, no turn on the spot.  None if it does not fit."""
    L = look_point(area, *start)
    sx, sy = float(start[0]), float(start[1])
    R = (float(L[0]) - sx) / 2.0
    if R < 0.15 or abs(float(L[1]) - sy) > 0.02:
        return None
    ph = np.linspace(math.pi, 2 * math.pi, max(8, int(math.pi * R / 0.0025)))
    return np.c_[sx + R + R * np.cos(ph), sy + R * np.sin(ph)]


def is_lanes(area) -> bool:
    return getattr(area, "main_shape", "ellipse") == "lanes"


def ends_facing_pad(area) -> bool:
    """True when the main path ends ON the look point facing the pad: the nose-in square (U-turn) or
    the lanes path (its last leg flies straight at the pad, nose along the path)."""
    return is_lanes(area) or (getattr(area, "main_shape", "ellipse") == "square"
                              and getattr(area, "main_nose", "out") == "in")


def lanes_waypoints(area) -> np.ndarray:
    """7 Nov 22:00 (the user's 2nd drawing): pad -> straight ahead past home tag 23 to the near row ->
    turn RIGHT along it to the right column -> up it -> across the top (past tag 28) -> down the LEFT
    column -> diagonal back to the apex on the centre line -> straight back to the look point (nose on
    the pad).  Columns 0.2 m outside the spot rows so a spot is beside the line (mid-picture), never
    under the drone."""
    yl, yr = area.lanes_y
    x0, x1 = area.lanes_x
    ax = area.apex_x
    L = look_point(area, 0.0, 0.0)
    return np.array([(0.0, 0.0), (x0, 0.0), (x0, yr), (x1, yr), (x1, yl), (x0, yl), (ax, 0.0),
                     (float(L[0]), float(L[1]))])


def turn_arc(Pa, A, Pb, R: float, step: float = 0.0025):
    """7 Nov 23:56: replace the corner A (coming from Pa, leaving towards Pb) by a circular arc of
    radius R that bends ONE way only.  The nose follows the path, so it turns smoothly and stops on
    the new direction - no loop, no overshoot (slow_turns draws a loop for turns > 100 deg).
    Returns (S, arc points, E): S on the way in, E on the way out."""
    Pa, A, Pb = (np.asarray(v, float) for v in (Pa, A, Pb))
    u1 = (A - Pa) / np.linalg.norm(A - Pa)
    u2 = (Pb - A) / np.linalg.norm(Pb - A)
    h1, h2 = math.atan2(u1[1], u1[0]), math.atan2(u2[1], u2[0])
    th = math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))           # + = clockwise seen from above
    T = R * math.tan(abs(th) / 2)
    S, E = A - T * u1, A + T * u2
    sg = 1.0 if th > 0 else -1.0
    C = S + sg * R * np.array([-math.sin(h1), math.cos(h1)])         # centre on the inside of the turn
    hs = np.linspace(h1, h1 + th, max(8, int(abs(th) * R / step)))
    arc = C - sg * R * np.c_[-np.sin(hs), np.cos(hs)]
    return S, arc, E


def lanes_path(area) -> np.ndarray:
    W = lanes_waypoints(area)
    # 7 Nov 23:56 (user's drawing): at the apex turn on the spot, CLOCKWISE, from the diagonal round to
    # facing the pad (56 -> 180 deg), along a tiny arc (radius apex_turn_r_m) - then straight to the
    # look point.  W[-3] = near-left corner, W[-2] = apex, W[-1] = look point.
    # The near-left corner (W[-3], 180 -> 56 deg, anticlockwise) gets the same kind of arc: one smooth
    # turn, no overshoot.
    R = float(getattr(area, "apex_turn_r", 0.12))
    S1, arc1, E1 = turn_arc(W[-4], W[-3], W[-2], R)
    S2, arc2, E2 = turn_arc(W[-3], W[-2], W[-1], R)

    def line(a, b, end=False):
        return np.linspace(a, b, max(2, int(np.hypot(*(np.asarray(b) - np.asarray(a))) / 0.005)), endpoint=end)
    pts = list(W[:-3]) + [S1]
    P = [line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return np.vstack(P + [arc1, line(E1, S2), arc2, line(E2, W[-1], end=True)])


def segment1(xB, yc, a, b, area):
    if is_lanes(area):
        # nose follows the path everywhere (no outward window): window = empty at the end
        Q = mt.slow_turns(lanes_path(area), turn_dps=TURN_DPS, speed=area.speed, end_hold=1.5, start_heading=0.0)
        return Q, (len(Q), len(Q))
    return _segment1_loop(xB, yc, a, b, area)


def _segment1_loop(xB, yc, a, b, area):
    """Pad -> T (nose forward, at tuning lead_speed_mps) -> hover tip_turn_s while the nose turns ->
    once round the ellipse (nose outward, top_hold_s at the far tip) -> T."""
    Pm = main_loop(xB, yc, a, b, area)
    sx, sy = float(Pm[0, 0]), float(Pm[0, 1])         # where the loop starts (dog bone: the near dent)
    lead = np.linspace((0.0, 0.0), (sx, sy), max(2, int(math.hypot(sx, sy) / 0.005)))
    Ql = mt.slow_turns(lead, turn_dps=TURN_DPS, speed=getattr(area, "lead_speed", area.speed), end_hold=0.0)
    face = ends_facing_pad(area)
    if face:
        # 7 Nov 19:10: nose IN: at the loop's start the nose (90 deg right of travel = +x) already points
        # the way the lead flew, so no turn there: set off along the loop at once, short hover only
        j = min(len(Pm) - 1, 20)
        h1 = math.atan2(Pm[j, 1] - sy, Pm[j, 0] - sx)
        Qe = mt.slow_turns(Pm, turn_dps=TURN_DPS, speed=area.speed, end_hold=0.0, start_heading=h1)
        hold_s = 0.5
    else:
        Qe = mt.slow_turns(Pm, turn_dps=TURN_DPS, speed=area.speed, end_hold=0.3,
                           start_heading=math.atan2(sy, sx))
        hold_s = getattr(area, "tip_turn_s", DWELL_S)
    k = len(Ql) - 1                                   # T: the nose starts turning outward here
    hold = np.repeat(Ql[-1:], int(round(hold_s / pc.DT)), axis=0)
    Q = np.vstack([Ql, hold, Qe[1:]])
    U = u_turn(area, (sx, sy)) if face else None
    if U is not None:                                 # the closing U-turn onto the look point
        tdps = float(getattr(area, "turn_home_dps", 20.0))
        Qu = mt.slow_turns(U, turn_dps=0.85 * tdps, speed=area.speed, end_hold=1.0, start_heading=-math.pi / 2)
        Q = np.vstack([Q, Qu[1:]])
    return _top_hold(Q, (k, len(Q)), getattr(area, "top_hold_s", TOP_HOLD_S))


def loop_cuts(Q, xB, yc, a, b, area) -> tuple[list, int | None]:
    """7 Nov 19:10: where to split the nose-in loop into pieces: just after each corner (the drone is
    on a straight then, the nose done turning), and where the closing U-turn starts.  Before each
    piece the anchor fix (run_baseline) may shift the rest of the path.  Returns (cuts, u_turn_start)."""
    if not ends_facing_pad(area):
        return [], None
    r = max(0.05, min(getattr(area, "fillet", 0.4), a - 0.01, b - 0.01))
    x0, x1, y0, y1 = xB, xB + 2 * a, yc - b, yc + b
    exits = [(x0 + r, y0), (x1, y0 + r), (x1 - r, y1), (x0, y1 - r)]     # going round nose-in (reversed)
    cuts, i = [], 0
    for p in exits:
        d = np.hypot(Q[i:, 0] - p[0], Q[i:, 1] - p[1])
        j = i + int(np.argmin(d))
        if d[j - i] < 0.02:
            cuts.append(j)
            i = j + 1
    sx, sy = xB, float(area.home_at[1])
    if not (yc - b + r <= sy <= yc + b - r):
        sy = yc
    near = np.where(np.hypot(Q[i:, 0] - sx, Q[i:, 1] - sy) < 0.003)[0]
    u0 = None
    if len(near) and u_turn(area, (sx, sy)) is not None:
        u0 = i + int(near[-1])
        cuts.append(u0)
    return cuts, u0


def segment2(xB, yc, a, b, area):
    """From T once round an adjusted ellipse through T (nose outward the whole way), slower."""
    P = ellipse(xB, yc, a, b)
    Q = mt.slow_turns(P, turn_dps=TURN_DPS, speed=area.search_speed, end_hold=0.3, start_heading=math.pi / 2)
    return _top_hold(Q, (0, len(Q)), getattr(area, "top_hold_s", TOP_HOLD_S))


def segment3(T, L, start_heading=None, speed=0.2):
    """T -> look point L, nose held (it already faces the pad).  The nose does not follow this path,
    so no slow 'turnaround' at the start (5 Nov: start_heading pi/2 added a 90 deg turn = ~7 s)."""
    T, L = np.asarray(T, float), np.asarray(L, float)
    if math.hypot(*(L - T)) < 0.02:                   # already there (6 Nov: 1.9 m square's tip = look point)
        return np.repeat(L[None, :], max(2, int(0.3 / pc.DT)), axis=0)
    if start_heading is None:
        start_heading = math.atan2(*(L - T)[::-1])
    P = np.linspace(T, L, max(2, int(math.hypot(*(L - T)) / 0.005)))
    return mt.slow_turns(P, turn_dps=25.0, speed=speed, end_hold=0.3, start_heading=start_heading)


def look_point(area, xB, yc):
    """Where to stop on the way back so tag 20 is ~look_dist ahead (nose on the pad)."""
    hx, hy = area.home_at
    # 7 Nov: ALWAYS straight ahead of the home tag (the landing that works), whatever loop is flown.
    # With the loop centred on the real area (y -0.225) its tip moved off the centre line and the
    # look point swung 0.6 m sideways - that would have changed the landing approach.
    d = np.array([1.0, 0.0])
    n = np.linalg.norm(d)
    if n < 1e-6:
        d, n = np.array([1.0, 0.0]), 1.0
    # 5 Nov: with a short run-out T can be closer to tag 20 than look_dist; then the look point is
    # a little further out than T (tag 20 too close would be under the camera's view)
    return np.array([hx, hy]) + d / n * area.look_dist


# ------------------------------------------------------------------ what the camera sees
def headings(Q, window, follow_outside=True, h0=0.0):
    """Nose the yaw v2 firmware flies: travel + 90 deg inside the window; outside it along the path
    (follow_outside) or held.  h0 = nose at the start of the segment."""
    rate, acc = mt.yaw_limits()
    yaw = np.full(len(Q), h0)
    h, w, want = h0, 0.0, h0
    for i in range(1, len(Q)):
        dx, dy = Q[i] - Q[i - 1]
        if dx * dx + dy * dy > 1e-10:
            if window[0] <= i - 1 < window[1]:
                want = mt._wrap(math.atan2(dy, dx) + math.pi / 2)
            elif follow_outside:
                want = mt._wrap(math.atan2(dy, dx))
        d = mt._wrap(want - h)
        vd = min(rate, math.sqrt(2 * acc * abs(d)))
        vd = math.copysign(min(vd, abs(d) / pc.DT), d)
        w += max(-acc * pc.DT, min(acc * pc.DT, vd - w))
        h = mt._wrap(h + w * pc.DT)
        yaw[i] = h
    return yaw


def frames(Q, Y, cam, pts, start_yaw=None):
    """Camera frames (at cam.fps) in which each point is in the usable view."""
    step = max(1, int(round(1.0 / (cam.fps * pc.DT))))
    idx = np.arange(0, len(Q), step)
    X, Yp, H = Q[idx, 0], Q[idx, 1], Y[idx]
    out = []
    for (tx, ty) in pts:
        dx, dy = tx - X, ty - Yp
        c, s = np.cos(H), np.sin(H)
        out.append(int(np.sum(pc.visible(dx * c + dy * s, -dx * s + dy * c, cam))))
    return out


def field_seen(Q, Y, cam, area, cell=0.1, need=3):
    gx = np.arange(area.x[0] + cell / 2, area.x[1], cell)
    gy = np.arange(area.y[0] + cell / 2, area.y[1], cell)
    G = np.array([(x, y) for x in gx for y in gy])
    f = np.array(frames(Q, Y, cam, G))
    return G, f >= need


# ------------------------------------------------------------------ planning
def drifted(Q, Y, win, d):
    """Where the drone ends up if it sits d metres the way its nose points inside the window
    (drone 5, D1 + D2 runs 3 Nov: PX4 track 0.2-0.3 m outside the ellipse, towards the nose).
    Builds up over the first 2 s of the window."""
    F = Q.copy()
    if d:
        k = np.clip((np.arange(len(Q) - win[0])) / (2.0 / pc.DT), 0.0, 1.0)[:, None]
        F[win[0]:] += d * k * np.c_[np.cos(Y[win[0]:]), np.sin(Y[win[0]:])]
    return F


def sent_and_flown(prm, area, seg=None):
    """For the ellipse we WANT flown (prm): what is SENT (pulled in by area.pull), its window and
    nose, and where the drone ends up with no drift and with area.drift (the two cases planned for).
    Returns Qsent, win, Ysent, [flown with drift 0, flown with drift area.drift]."""
    seg = seg or segment1
    c = pulled(prm, area.pull)
    Qs, win = seg(c["xB"], c["yc"], c["a"], c["b"], area)
    Ys = headings(Qs, win, h0=0.0 if seg is segment1 else math.pi)
    ds = (0.0, area.drift) if area.drift else (0.0,)
    return Qs, win, Ys, [drifted(Qs, Ys, win, d) for d in ds]


def plan_first(area, cam=None, verbose=False):
    """Ellipse 1: every spot and the anchor seen (most frames for the worst one) BOTH if the drone
    flies the ellipse exactly and if it drifts area.drift m outward, then the shortest.  Narrow
    ellipses (from 0.3 m wide) are allowed.  T at least look_dist + 0.1 m beyond tag 20."""
    cam = cam or mt.camera_model(area.height)
    anc = list(getattr(area, "anchors", {area.anchor_id: area.anchor_at}).values())
    ns = len(area.spots)
    pts = list(area.spots) + anc
    if is_lanes(area):                                # 7 Nov 21:30: fixed path, nothing to choose
        yl, yr = area.lanes_y
        x0, x1 = area.lanes_x
        p = dict(xB=float(x0), yc=float((yl + yr) / 2), a=float((x1 - x0) / 2), b=float((yr - yl) / 2))
        Q, win = segment1(0, 0, 0, 0, area)
        f = frames(Q, headings(Q, win), cam, pts)
        if verbose:
            print("lanes:", p, "frames per spot/anchor:", f, f"{len(Q) * pc.DT:.0f} s")
        return p, Q, win, f
    best = None
    fixed = getattr(area, "main_loop", None)
    if fixed and getattr(area, "main_centre", None) is not None:   # tuning main_centre_m: loop centred there
        xs = np.array([area.main_centre[0] - fixed[0] / 2])
    elif getattr(area, "tip_x", None) is not None:      # tuning: tip_x_m fixes the run-out length
        xs = np.array([float(area.tip_x)])
    else:                                             # tuning: tip_x_min_m .. tip_x_max_m
        xs = np.arange(getattr(area, "tip_x_min", area.x[0] + 0.5), getattr(area, "tip_x_max", 1.6) + 1e-6, 0.1)
    enough = getattr(area, "enough_frames", ENOUGH_FRAMES)
    ycs = np.arange(-0.10, 0.101, 0.05) + (area.y[0] + area.y[1]) / 2
    if fixed and getattr(area, "main_centre", None) is not None:
        ycs = np.array([area.main_centre[1]])
    for xB in xs:
        for yc in ycs:
            fixed = getattr(area, "main_loop", None)      # tuning main_loop_m: size fixed, planner picks T
            for a in ([fixed[0] / 2] if fixed else np.arange(0.5, 1.01, 0.1)):
                m_edge = -0.30 if fixed else 0.2      # 7 Nov: a FIXED loop may run on / just outside the edge (20:05: up to 0.3 m, 4.1 x 3.35)
                if xB + 2 * a > area.x[1] - m_edge:
                    continue
                for b in ([fixed[1] / 2] if fixed else np.arange(0.15, 0.501, 0.05)):
                    if yc - b < area.y[0] + m_edge or yc + b > area.y[1] - m_edge:
                        continue
                    p = dict(xB=float(xB), yc=float(yc), a=float(a), b=float(b))
                    Qs, win, Ys, flown = sent_and_flown(p, area)
                    if len(Qs) > 6 * mt.MAX_POINTS:      # 5 Nov (3): flown in pieces now
                        continue
                    fs = [frames(F, Ys, cam, pts) for F in flown]
                    # every spot and at least ONE map anchor well seen (must); then ALL the map
                    # anchors well seen if any ellipse manages it (5 Nov: costs ~2 s); then fastest
                    worst = min(min(min(g[:ns]), max(g[ns:])) for g in fs)
                    all_anchors = min(min(g[ns:]) for g in fs)
                    score = (min(worst, enough), min(all_anchors, enough), -len(Qs), worst)
                    if best is None or score > best[0]:
                        k = -1 if getattr(area, "expect_drift", area.drift) else 0
                        best = (score, p, flown[k], win, fs[k])          # what we expect it to fly
    if best is None:
        raise ValueError("no ellipse fits the search area")
    _, prm, Q, win, f = best
    if verbose:
        print("ellipse 1:", {k: round(v, 3) for k, v in prm.items()}, "frames per spot/anchor:", f,
              f"{len(Q) * pc.DT:.0f} s")
    return prm, Q, win, f


def plan_retry(area, prm, found_xy, cam=None, seen_before=None, anchor_missing=False, too_many=False):
    fixed = getattr(area, "backup_loop", None)
    if fixed:                                     # tuning backup_loop_m: this size, through the same tip T
        cam = cam or mt.camera_model(area.height)
        xB, yc = prm["xB"], prm["yc"]
        left = [s for s in area.spots if all(math.hypot(s[0] - x, s[1] - y) > 0.45 for x, y in found_xy)]
        if anchor_missing:
            anc = list(getattr(area, "anchors", {area.anchor_id: area.anchor_at}).values())
            mid = (prm["xB"] + prm["a"], prm["yc"])
            left = left + [min(anc, key=lambda q: math.hypot(q[0] - mid[0], q[1] - mid[1]))]
        bx = getattr(area, "backup_tip_x", None)      # 5 Nov (2): backup centred in the area, own tip
        p = dict(xB=float(bx if bx is not None else xB), yc=float(yc), a=fixed[0] / 2, b=fixed[1] / 2)
        Qs, win, Y, flown = sent_and_flown(p, area, segment2)
        return p, flown[-1 if getattr(area, "expect_drift", area.drift) else 0], win, left
    """Adjusted ellipse through the same tip T, slower, aimed (most frames for the worst one) at:
    the spots with no confirmed tag near them, plus the anchor if it was not seen; with too many
    targets (a wrong decode) at every spot so the real tags pile up frames; else at the ground the
    first ellipse did not see."""
    cam = cam or mt.camera_model(area.height)
    xB, yc = prm["xB"], prm["yc"]
    if too_many:
        left = list(area.spots)
    else:
        left = [s for s in area.spots if all(math.hypot(s[0] - x, s[1] - y) > 0.45 for x, y in found_xy)]
    if anchor_missing:                    # aim at the map anchor nearest the middle of ellipse 1
        anc = list(getattr(area, "anchors", {area.anchor_id: area.anchor_at}).values())
        mid = (prm["xB"] + prm["a"], prm["yc"])
        left = left + [min(anc, key=lambda q: math.hypot(q[0] - mid[0], q[1] - mid[1]))]
    best = None
    for a in np.arange(0.4, 1.41, 0.2):
        if xB + 2 * a > area.x[1] - 0.1:
            continue
        for b in np.arange(0.2, 0.91, 0.14):
            if yc - b < area.y[0] + 0.1 or yc + b > area.y[1] - 0.1:
                continue
            if abs(a - prm["a"]) < 0.05 and abs(b - prm["b"]) < 0.05:
                continue                                  # must differ from the first one
            p = dict(xB=xB, yc=yc, a=float(a), b=float(b))
            Qs, win, Y, flown = sent_and_flown(p, area, segment2)
            if len(Qs) > 3 * mt.MAX_POINTS:      # 5 Nov (3): flown in pieces now
                continue
            F = flown[-1]                                 # with the drift it must still stay inside
            if F[:, 0].max() > area.x[1] - 0.05 or F[:, 1].min() < area.y[0] + 0.05 or F[:, 1].max() > area.y[1] - 0.05:
                continue
            if left:
                fs = [frames(F, Y, cam, left) for F in flown]
                score = (min(min(f) for f in fs), min(sum(f) for f in fs), -len(Qs))
            else:
                seen = np.ones(1, bool)
                for F in flown:
                    G, s1 = field_seen(F, Y, cam, area)
                    seen = seen & s1
                new = seen & ~seen_before if seen_before is not None else seen
                score = (int(new.sum()), 0, -len(Qs))
            if best is None or score > best[0]:
                best = (score, p, flown[-1 if getattr(area, "expect_drift", area.drift) else 0], win)
    if best is None:
        return None
    return best[1], best[2], best[3], left


# ------------------------------------------------------------------ preview
def draw(area, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    cam = mt.camera_model(area.height)
    prm, Q1, w1, f1 = plan_first(area, cam)
    Y1 = headings(Q1, w1)
    G, seen1 = field_seen(Q1, Y1, cam, area)
    L = look_point(area, prm["xB"], prm["yc"])
    Q3 = segment3((prm["xB"], prm["yc"]), L)
    # example backup: pretend the spot with the fewest frames was missed
    worst = [area.spots[int(np.argmin(f1[:len(area.spots)]))]] if area.spots else []
    found = [s for s in area.spots if s not in worst]
    r = plan_retry(area, prm, found, cam, seen1)
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 8.6))
    for ax, show_retry in zip(axs, (False, True)):
        ax.add_patch(Rectangle((area.y[0], area.x[0]), area.y[1] - area.y[0], area.x[1] - area.x[0], fill=False,
                               lw=1.4, ec="#1f2328"))
        ax.add_patch(Rectangle((-0.45, -0.45), 0.9, 0.9, fc="#7cb36b", ec="#1f2328", alpha=.5))
        ax.text(0.5, -0.3, "pad 0.9 x 0.9", fontsize=8)
        if not show_retry:
            ax.scatter(G[:, 1], G[:, 0], c=np.where(seen1, "#cfe3c7", "#f3f3f3"), s=34, marker="s", lw=0, zorder=0)
            k0 = w1[0]
            ax.plot(Q1[:k0, 1], Q1[:k0, 0], color="#2f6fb3", lw=2.2, label="1 out: nose forward")
            ax.plot(Q1[k0:, 1], Q1[k0:, 0], color="#d9822b", lw=2.2, label="1 ellipse: nose outward")
            idx = np.arange(k0 + int(area.tip_turn_s / pc.DT), len(Q1), int(1.5 / pc.DT))
            ax.quiver(Q1[idx, 1], Q1[idx, 0], np.sin(Y1[idx]), np.cos(Y1[idx]), color="#d9822b", scale=9,
                      width=.005, zorder=5)
            if area.pull > 0:
                c = pulled(prm, area.pull)
                E = main_loop(c["xB"], c["yc"], c["a"], c["b"], area)
                ax.plot(E[:, 1], E[:, 0], color="#8a5a00", lw=1.0, ls=":",
                        label=f"sent to the drone (pulled in {area.pull:.2f} m)")
            ax.plot(Q3[:, 1], Q3[:, 0], color="#3a8a5c", lw=2.2, ls="--", label="3 back: nose on the pad")
            ax.annotate("", xy=(0, 0), xytext=(L[1], L[0]), arrowprops=dict(arrowstyle="-|>", color="#3a8a5c",
                                                                              lw=1.6, ls=":"))
            ax.plot(L[1], L[0], "o", color="#3a8a5c", ms=7)
            ax.annotate(f"look at tag {area.home_id} here,\nthen fly onto the pad", (L[1], L[0]), textcoords="offset points",
                        xytext=(10, -4), fontsize=8)
            T1 = len(Q1) * pc.DT
            ax.set_title(f"Normal flight: {dict(square='rounded square', dogbone='dog bone', lanes='lanes (nose along the path)').get(area.main_shape, 'ellipse')} {2 * prm['a']:.1f} x {2 * prm['b']:.1f} m, {T1:.0f} s "
                         f"(+ way back)\ngreen = ground the camera sees ({100 * seen1.mean():.0f}% of the area)",
                         fontsize=10)
        else:
            ax.plot(Q1[w1[0]:, 1], Q1[w1[0]:, 0], color="#d9822b", lw=1.2, alpha=.5, label="1 ellipse")
            if r:
                prm2, Q2, w2, left = r
                Y2 = headings(Q2, w2, h0=math.pi)
                ax.plot(Q2[:, 1], Q2[:, 0], color="#8e44ad", lw=2.2, label="2 backup ellipse (slower)")
                idx = np.arange(0, len(Q2), int(1.5 / pc.DT))
                ax.quiver(Q2[idx, 1], Q2[idx, 0], np.sin(Y2[idx]), np.cos(Y2[idx]), color="#8e44ad", scale=9,
                          width=.005, zorder=5)
                ax.set_title(f"Only if a target is missing: backup ellipse (centred)\n"
                             f"example: spot {tuple(round(v, 2) for v in worst[0]) if worst else '-'} missed -> "
                             f"{2 * prm2['a']:.1f} x {2 * prm2['b']:.1f} m, {len(Q2) * pc.DT:.0f} s, aimed at it",
                             fontsize=10)
        for i, (sx, sy) in enumerate(area.spots):
            ax.plot(sy, sx, "o", mfc="white", mec="#b42318", ms=9, mew=1.5, zorder=7)
            if not show_retry:
                ax.annotate(f"{f1[i]}", (sy, sx), textcoords="offset points", xytext=(7, 4), fontsize=7.5,
                            color="#b42318")
        for _aid, _at in area.anchors.items():
            if _aid == area.anchor_id:
                continue
            ax.plot(_at[1], _at[0], "s", color="#1f2328", ms=9, zorder=7)
            ax.annotate(f"anchor {_aid}", (_at[1], _at[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.plot(area.anchor_at[1], area.anchor_at[0], "s", color="#1f2328", ms=9, zorder=7)
        ax.annotate(f"anchor {area.anchor_id}", (area.anchor_at[1], area.anchor_at[0]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5)
        ax.plot(area.home_at[1], area.home_at[0], "s", color="#1f2328", ms=9, zorder=7)
        ax.annotate(f"home {area.home_id}", (area.home_at[1], area.home_at[0]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xlim(area.y[0] - 0.4, area.y[1] + 0.4)
        ax.set_ylim(-0.6, area.x[1] + 0.3)
        ax.grid(alpha=.2)
        ax.set_xlabel("right of the pad centre (m)")
        ax.set_ylabel("ahead of the pad centre (m)")
        ax.legend(fontsize=8, loc="upper right", framealpha=.9)
    fig.suptitle("Search mission. Red circles = possible tag spots (number = camera frames on it); "
                 "arrows = nose every 1.5 s", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return prm, Q1, f1


if __name__ == "__main__":
    area = load_area()
    out = HERE / "trajectories" / "search_plan.png"
    out.parent.mkdir(exist_ok=True)
    prm, Q1, f1 = draw(area, out)
    print(f"ellipse 1: tip T ({prm['xB']:.2f}, {prm['yc']:+.2f}), {2 * prm['a']:.2f} m long x {2 * prm['b']:.2f} m wide, "
          f"{len(Q1) * pc.DT:.0f} s; frames on each spot / anchor: {f1}")
    print(f"picture: {out}")
