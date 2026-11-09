"""Coverage path planner: paths whose camera footprint sweeps a rectangular field.

  python plan_coverage.py --length 3.0 --width 2.0            (field in metres)

Field (start frame): x = 0 .. LENGTH straight ahead of the drone's nose at the start mark,
y = -WIDTH/2 .. +WIDTH/2 (right = +).  The drone starts and lands at (0, 0).

Camera model (from config / the 26 Oct calibration): pinhole fx 183.3 fy 182.9 at 320x240,
pitched --pitch deg below horizontal at --height m, a tag only counts within --max-range m of
the camera and away from the image border (--margin-px).  A ground cell is DETECTED when it is
inside the footprint in at least --min-frames frames at --fps (the 3-frame confirmation rule).

Three plans are built and scored (coverage %, overlap, path length, time):
  lawnmower      wave: lanes + half-circle U-turns, fixed heading (works on today's firmware)
  lawnmower_yaw  same wave, nose TANGENT to the path (needs the yaw-follow firmware patch)
  ellipse        fixed heading (works on today's firmware)
  ellipse_yaw    nose TANGENT to the ellipse, one full turn of heading per lap
                 (needs the yaw-follow firmware patch + run_baseline.py --yaw-follow)

Each plan is searched over its size parameters for the SHORTEST path that reaches --target
coverage (falls back to the best coverage if none does).  Outputs in --out:
  coverage_<plan>.csv   t,x,y,z,yaw_rad  (yaw = camera heading, 0 = start nose, + = right)
  coverage_<plan>.png   path, camera arrows, footprint wedges, detected cells
  coverage_summary.txt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

FX, FY, CX, CY, W_PX, H_PX = 183.29, 182.86, 161.04, 120.12, 320, 240
DT = 0.05


# ---------------------------------------------------------------------------- footprint
def visible(f, r, a):
    """Ground points (f ahead, r right of the drone, body frame) seen well enough by the camera."""
    p = math.radians(a.pitch)
    c, s = math.cos(p), math.sin(p)
    fc = f - a.cam_fwd
    h = a.height
    xc = r
    yc = -s * fc + c * h
    zc = c * fc + s * h
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * xc / zc + CX
        v = FY * yc / zc + CY
    rng = np.sqrt(fc * fc + r * r + h * h)
    m = a.margin_px
    return (zc > 0.05) & (u >= m) & (u <= W_PX - m) & (v >= m) & (v <= H_PX - m) & (rng <= a.max_range)


def footprint_extent(a):
    f = np.linspace(0, 3, 601)
    r = np.linspace(-2, 2, 801)
    F, R = np.meshgrid(f, r)
    vis = visible(F, R, a)
    fs, rs = F[vis], R[vis]
    near, far = fs.min(), fs.max()
    mid = 0.5 * (near + far)
    band = np.abs(F - mid) < 0.02
    half_w = np.abs(R[vis & band]).max() if np.any(vis & band) else 0.3
    return near, far, half_w


# ---------------------------------------------------------------------------- timing
def time_path(waypts, stops, a, end_hold=2.0):
    """Polyline (dense) with rest-to-rest legs between stop indices -> samples every DT."""
    out = [waypts[:1]]
    for i0, i1 in zip(stops[:-1], stops[1:]):
        seg = waypts[i0:i1 + 1]
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        L = d[-1]
        if L < 1e-6:
            continue
        ta = a.speed / a.accel
        da = 0.5 * a.accel * ta * ta
        if 2 * da > L:
            ta = math.sqrt(L / a.accel); vm = a.accel * ta; da = L / 2
        else:
            vm = a.speed
        T = 2 * ta + (L - 2 * da) / vm
        t = np.arange(DT, T + 1e-9, DT)
        sd = np.clip(np.where(t < ta, 0.5 * a.accel * t ** 2,
                              np.where(t < T - ta, da + vm * (t - ta), L - 0.5 * a.accel * (T - t) ** 2)), 0, L)
        out.append(np.c_[np.interp(sd, d, seg[:, 0]), np.interp(sd, d, seg[:, 1])])
    P = np.vstack(out)
    P = np.vstack([P, np.repeat(P[-1:], int(end_hold / DT), axis=0)])
    return P


def heading_of(P, yaw_follow):
    if not yaw_follow:
        return np.zeros(len(P))
    d = np.diff(P, axis=0, prepend=P[:1])
    yaw = np.zeros(len(P))
    last = 0.0
    for i, (dx, dy) in enumerate(d):
        if math.hypot(dx, dy) > 1e-4:
            last = math.atan2(dy, dx)
        yaw[i] = last
    return yaw


# ---------------------------------------------------------------------------- scoring
def score(P, yaw, a, grid):
    """coverage = share of field cells (beyond the start strip) seen in >= min_frames frames;
    overlap = mean number of separate PASSES over each detected cell (1.0 = no overlap)."""
    gx, gy = grid
    cnt = np.zeros(gx.shape, dtype=np.int32)
    passes = np.zeros(gx.shape, dtype=np.int32)
    prev = np.zeros(gx.shape, dtype=bool)
    step = max(1, int(round(1.0 / (a.fps * DT))))
    for i in range(0, len(P), step):
        x, y, h = P[i, 0], P[i, 1], yaw[i]
        c, s = math.cos(h), math.sin(h)
        dx, dy = gx - x, gy - y
        vis = visible(dx * c + dy * s, -dx * s + dy * c, a)
        cnt += vis
        passes += vis & ~prev
        prev = vis
    field = gx >= a.start_strip
    det = (cnt >= a.min_frames) & field
    cov = det.sum() / field.sum()
    over = passes[det].mean() if det.any() else 0.0
    L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
    return cov, over, L, cnt


# ---------------------------------------------------------------------------- plans
def plan_lawnmower(a, near, far, hw, spacing):
    """Lanes straight ahead/back at fixed heading, spaced `spacing` apart across the field."""
    W, Lf = a.width, a.length
    n = max(1, math.ceil(W / spacing - 1e-9))
    ys = (np.arange(n) - (n - 1) / 2) * spacing
    x_top = max(0.3, Lf - far * 0.9)          # the camera sees `far` ahead: stop short of the far edge
    pts = [(0.0, 0.0)]
    x = 0.0
    for y in ys:
        pts.append((x, y))                      # sideways to the lane
        x = x_top if x == 0.0 else 0.0
        pts.append((x, y))                      # along the lane
    if x != 0.0:
        pts.append((0.0, ys[-1]))
    pts.append((0.0, 0.0))
    pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    return densify(np.array(pts, float), set(range(len(pts))))


def plan_serpentine(a, across, spacing, end_margin, x_first, x_last=None):
    """Wave-shaped lawnmower: straight lanes joined by half-circle U-turns (radius = spacing/2).

    across=True : lanes run left<->right (along y), stacked ahead in x from x_first.
    across=False: lanes run ahead<->back (along x), stacked across y.
    Joined to the start (0,0) at both ends.  Returns (dense points, stop indices)."""
    W, Lf = a.width, a.length
    r = spacing / 2
    pts = []
    if across:
        y_lo, y_hi = -W / 2 + end_margin, W / 2 - end_margin
        if y_hi - y_lo < 0.2:
            return None
        x_end = x_last if x_last is not None else Lf - end_margin
        xs = [x_first]
        while xs[-1] + spacing <= x_end + 1e-9:
            xs.append(xs[-1] + spacing)
        go_right = False                               # first lane right-to-left? start on the left end
        cur = np.array([xs[0], y_hi if not go_right else y_lo])
        for i, x in enumerate(xs):
            ya, yb = (y_lo, y_hi) if go_right else (y_hi, y_lo)
            pts.append(np.linspace([x, ya], [x, yb], max(2, int(abs(yb - ya) / 0.01))))
            if i < len(xs) - 1:                        # U-turn towards +x
                th = np.linspace(0, math.pi, 60)
                cy, cx = yb, x + r
                sgn = 1 if go_right else -1            # bulge outwards past the lane end
                pts.append(np.c_[cx - r * np.cos(th), cy + sgn * r * np.sin(th)])
            go_right = not go_right
    else:
        x_lo, x_hi = x_first, (x_last if x_last is not None else Lf - end_margin)
        if x_hi - x_lo < 0.2:
            return None
        n = max(1, int(math.floor((W - 2 * end_margin) / spacing + 1e-9)) + 1)
        ys = (np.arange(n) - (n - 1) / 2) * spacing
        up = True
        for i, y in enumerate(ys):
            xa, xb = (x_lo, x_hi) if up else (x_hi, x_lo)
            pts.append(np.linspace([xa, y], [xb, y], max(2, int(abs(xb - xa) / 0.01))))
            if i < len(ys) - 1:
                th = np.linspace(0, math.pi, 60)
                sgn = 1 if up else -1
                pts.append(np.c_[xb + sgn * r * np.sin(th), y + r - r * np.cos(th)])
            up = not up
    body = np.vstack(pts)
    lead_in = np.linspace([0.0, 0.0], body[0], 60)
    lead_out = np.linspace(body[-1], [0.0, 0.0], 60)
    dense = np.vstack([lead_in, body[1:], lead_out[1:]])
    stops = [0, len(lead_in) - 1, len(lead_in) - 1 + len(body) - 1, len(dense) - 1]
    return dense, stops


def densify(pts, stops, ds=0.01):
    dense, idx = [pts[0]], [0]
    for i in range(1, len(pts)):
        seg = np.linspace(pts[i - 1], pts[i], max(2, int(np.hypot(*(pts[i] - pts[i - 1])) / ds) + 1))[1:]
        dense.extend(seg)
        if i in stops:
            idx.append(len(dense) - 1)
    return np.array(dense), idx


def plan_ellipse(a, cx, ay, by, turns=1.0, shrink=0.0):
    """Ellipse (or elliptical spiral) through its near end, joined to the start (0,0)."""
    n = int(2000 * turns)
    th = np.linspace(math.pi, math.pi - 2 * math.pi * turns, n)
    k = np.linspace(0, 1, n)
    A = ay - shrink * k
    B = by - shrink * k
    path = np.c_[cx + A * np.cos(th), -B * np.sin(th)]
    lead_in = np.linspace([0.0, 0.0], path[0], 50)
    lead_out = np.linspace(path[-1], [0.0, 0.0], 50)
    dense = np.vstack([lead_in, path[1:], lead_out[1:]])
    stops = [0, len(lead_in) - 1, len(lead_in) - 1 + len(path) - 1, len(dense) - 1]
    return dense, stops


def search(a, kind, grid, near, far, hw):
    best = None
    cands = []
    if kind in ("lawnmower", "lawnmower_yaw"):
        yaw = kind == "lawnmower_yaw"
        for across in (True,):                   # left-right lanes stepping away from the start
            for sp in np.arange(0.4, 1.61, 0.15):
                for em in np.arange(0.0, 0.61, 0.15):
                    for xf in np.arange(0.0, 0.91, 0.15):
                        if across:
                            xl = a.length - (em if yaw else max(em, far * 0.9))
                            lanes_n = int(math.floor((xl - xf) / sp + 1e-9)) + 1
                        else:
                            xl = a.length - (em if yaw else far * 0.9)
                            lanes_n = int(math.floor((a.width - 2 * em) / sp + 1e-9)) + 1
                        if lanes_n < a.min_lanes:
                            continue
                        res = plan_serpentine(a, across, sp, em, xf, xl)
                        if res is None:
                            continue
                        dn = res[0]                      # must stay inside the field (walls)
                        if (dn[:, 0].min() < -0.01 or dn[:, 0].max() > a.length + 0.01
                                or np.abs(dn[:, 1]).max() > a.width / 2 + 0.01):
                            continue
                        lanes = "left-right" if across else "ahead-back"
                        cands.append((f"{lanes_n} {lanes} lanes {sp:.2f} m apart", res))
    else:
        yaw = kind == "ellipse_yaw"
        Lf, W = a.length, a.width
        for inset_x in np.arange(0.0, 1.21, 0.1):
            for inset_y in np.arange(0.0, 0.91, 0.1):
                for turns, shrink in ((1.0, 0.0), (1.5, 0.4), (2.0, 0.5)):
                    if yaw:
                        x0 = 0.0
                        x1 = Lf - inset_x
                    else:                       # fixed heading: camera looks ahead -> stop short
                        x0 = 0.0
                        x1 = Lf - inset_x - near
                    ay = (x1 - x0) / 2
                    by = W / 2 - inset_y
                    if ay < 0.3 or by < 0.2 or ay - shrink < 0.2 or by - shrink < 0.15:
                        continue
                    cands.append(("inset %.1f/%.1f m, %.1f turns" % (inset_x, inset_y, turns),
                                  plan_ellipse(a, x0 + ay, ay, by, turns, shrink)))
    for label, (dense, stops) in cands:
        P = time_path(dense, stops, a)
        yawv = heading_of(P, kind.endswith("_yaw"))
        cov, over, L, cnt = score(P, yawv, a, grid)
        key = (cov >= a.target, -L if cov >= a.target else cov)
        if best is None or key > best[0]:
            best = (key, label, P, yawv, cov, over, L, cnt)
    return best


# ---------------------------------------------------------------------------- output
def write_csv(path, P, yaw):
    with open(path, "w") as f:
        f.write("t,x,y,z,yaw_rad\n")
        for i, ((x, y), h) in enumerate(zip(P, yaw)):
            f.write(f"{i * DT:.3f},{x:.4f},{y:.4f},0.0000,{h:.4f}\n")


def draw(path, kind, label, P, yaw, cnt, grid, a, cov, over, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    gx, gy = grid
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    det = cnt >= a.min_frames
    ax.pcolormesh(gy, gx, np.where(det, 1.0, np.nan), cmap="Greens", vmin=0, vmax=2.2, shading="auto")
    ax.add_patch(Rectangle((-a.width / 2, 0), a.width, a.length, fill=False, lw=1.5, ec="k"))
    ax.plot(P[:, 1], P[:, 0], "-", color="C0", lw=1.6, label="path")
    # footprint wedges + camera arrows every ~2 s
    f = np.linspace(0, 2.5, 120)
    r = np.linspace(-1.6, 1.6, 160)
    F, R = np.meshgrid(f, r)
    vis = visible(F, R, a)
    idx = np.arange(0, len(P), int(2.0 / DT))
    for j, i in enumerate(idx):
        x, y, h = P[i, 0], P[i, 1], yaw[i]
        c, s = math.cos(h), math.sin(h)
        if j % 3 == 0:
            pts = np.c_[F[vis], R[vis]]
            from scipy.spatial import ConvexHull
            hull = pts[ConvexHull(pts).vertices]
            wx = x + hull[:, 0] * c - hull[:, 1] * s
            wy = y + hull[:, 0] * s + hull[:, 1] * c
            ax.add_patch(Polygon(np.c_[wy, wx], closed=True, fc="C1", ec="C1", alpha=0.10, lw=0.6))
        ax.annotate("", xy=(y + 0.28 * s, x + 0.28 * c), xytext=(y, x),
                    arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.3))
    ax.plot(0, 0, "r*", ms=13, label="start / land")
    ax.plot([], [], color="C3", lw=1.3, label="camera heading (every 2 s)")
    ax.add_patch(Polygon([[0, 0]], fc="C1", alpha=0.2, label="camera footprint (sampled)"))
    ax.set_xlim(-a.width / 2 - 0.9, a.width / 2 + 0.9)
    ax.set_ylim(-0.5, a.length + 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.set_xlabel("right of start (m)")
    ax.set_ylabel("ahead of start (m)")
    need = "  [needs --yaw-follow firmware]" if kind.endswith("_yaw") else ""
    ax.set_title(f"{kind}: {label}{need}\ncoverage {cov * 100:.0f}%  overlap x{over:.1f}  "
                 f"path {L:.1f} m  time {len(P) * DT:.0f} s", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=3.0, help="field length straight ahead (m)")
    ap.add_argument("--width", type=float, default=2.0, help="field width, centred on the start (m)")
    ap.add_argument("--height", type=float, default=0.55, help="camera height above the floor (m)")
    ap.add_argument("--pitch", type=float, default=25.0, help="camera pitch below horizontal (deg)")
    ap.add_argument("--cam-fwd", type=float, default=0.02)
    ap.add_argument("--max-range", type=float, default=1.2, help="tag gate max_range_m (m)")
    ap.add_argument("--margin-px", type=float, default=20.0, help="ignore this close to the image border")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--accel", type=float, default=0.12)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.95, help="required share of the field detected")
    ap.add_argument("--start-strip", type=float, default=0.45,
                    help="first metres ahead of the start mark not counted (the drone stands there; "
                         "a forward-looking camera cannot see it from the start)")
    ap.add_argument("--plans", default="lawnmower,lawnmower_yaw,ellipse,ellipse_yaw")
    ap.add_argument("--min-lanes", type=int, default=3, help="lawnmower: at least this many lanes")
    ap.add_argument("--out", default="coverage_plans")
    a = ap.parse_args()

    near, far, hw = footprint_extent(a)
    print(f"camera footprint: {near:.2f}-{far:.2f} m ahead, half-width {hw:.2f} m at mid range")
    gx, gy = np.meshgrid(np.arange(0.025, a.length, 0.05), np.arange(-a.width / 2 + 0.025, a.width / 2, 0.05),
                         indexing="ij")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"field {a.length} x {a.width} m, height {a.height} m, pitch {a.pitch} deg, "
             f"max range {a.max_range} m, target {a.target * 100:.0f}% detected (>= {a.min_frames} frames), "
             f"first {a.start_strip} m ahead of the start not counted; overlap = passes per detected cell"]
    for kind in a.plans.split(","):
        best = search(a, kind, (gx, gy), near, far, hw)
        _, label, P, yawv, cov, over, L, cnt = best
        write_csv(out / f"coverage_{kind}.csv", P, yawv)
        draw(out / f"coverage_{kind}.png", kind, label, P, yawv, cnt, (gx, gy), a, cov, over, L)
        line = (f"{kind:12s} {label:28s} coverage {cov * 100:5.1f}%  passes/cell {over:4.2f}  "
                f"path {L:5.1f} m  time {len(P) * DT:5.1f} s  "
                f"x {P[:, 0].min():.2f}..{P[:, 0].max():.2f}  y {P[:, 1].min():+.2f}..{P[:, 1].max():+.2f}")
        print(line)
        lines.append(line)
    (out / "coverage_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/coverage_*.csv / .png / coverage_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
