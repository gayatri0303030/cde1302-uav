"""Make a trajectory CSV from the size of the area, for run_baseline.py --trajectory.

  python make_trajectory.py ellipse 5 2                  ellipse 5 m long (ahead) x 2 m wide
  python make_trajectory.py rectangle 5 2                the box outline, corners rounded
  python make_trajectory.py lawnmower_vertical 5 2       lanes running ahead/back (--lanes, default 4)
  python make_trajectory.py lawnmower_horizontal 5 2     lanes running left/right, stepping ahead
  python make_trajectory.py outward_rectangle 5 3        FIELD 5 x 3: small loop inside it, camera facing out

  options: --speed 0.2 (m/s)  --accel 0.12 (m/s^2)  --lanes N  --out FILE.csv  --no-plot

run_baseline.py does the same by itself:  --trajectory ellipse 5 2  (the CSV and a picture of it
are saved in trajectories\\ and the run uses that file).  Add --yaw-follow to the run for the
nose to turn along the path; the CSV is the same either way.

The box: LENGTH straight ahead of the start nose (x = 0 .. LENGTH) and WIDTH across, centred on
the start circle (y = -WIDTH/2 .. +WIDTH/2), i.e. the start circle sits in the middle of the near
edge, like the 2 Nov experiment box (2.3 x 1.5).  Every path touches the edges of the box, starts
and ends on the start circle, pauses 1 s there, then creeps 0.15 m ahead slowly so the nose faces
the home tag again for the landing (the yaw-follow firmware keeps the last heading).
z = 0 = the take-off hover; run_baseline --height adds the climb.

outward_rectangle (needs the yaw v2 firmware patch): LENGTH x WIDTH is the FIELD.  The drone flies
forward from wherever it hovers after looking at tag 20 to the near-right corner of a small rounded
rectangle centred in the field, goes once round it with the nose 90 deg right of travel (= always
facing OUT of the loop), stops at the middle of the loop's near side (nose already on the start)
and flies straight back, nose on the start, then lands at once (no creep, no homing).  It slows
down in the corners so the nose (config.yaml yaw: limits) can keep up.  The loop is the smallest
one whose camera coverage of the field is within 1 % of the best loop (camera model = the mount in
mapper_config_current.yaml at --height, default 0.8 m).  Writes <name>.json beside the CSV: the
points where "nose outward" is on, read by run_baseline.py.

Limits: the drone takes at most 2400 points = 120 s per trajectory (0.05 s apart) and 0.30 m/s
(config.yaml flight.max_speed_mps).  A path that is too long is refused with the speed that would
make it fit, or fly fewer lanes.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan_coverage as pc                                    # noqa: E402
import plan_experiment as pe                                  # noqa: E402

SHAPES = ("ellipse", "rectangle", "lawnmower_vertical", "lawnmower_horizontal", "outward_rectangle")
MAX_POINTS = 2400                                             # TRAJ_MAX_PTS in protocol.py / the firmware
MAX_SPEED = 0.30


# ------------------------------------------------------------------ outward-facing loop
R_CORNER = 0.25                                               # corner radius; speed drops in corners to suit the yaw limit
END_HOLD_S = 0.5                                              # outward loop: land straight after this


def camera_model(height: float = 0.8, speed: float = 0.2, accel: float = 0.12):
    """The flights' camera (mapper_config_current.yaml mount, config.yaml range rule) for coverage."""
    from miniproject.settings import load_settings, mount_from_mapper
    opt = HERE / "optimization_comparison" / "laptop"
    mfile = next(p for p in (opt / "mapper_config_current.yaml", opt / "mapper_config_29_Oct.yaml") if p.exists())
    m = mount_from_mapper(mfile)
    rng = load_settings(str(HERE / "miniproject" / "config.yaml")).gate.max_range_m + max(0.0, height - 0.5) / 3
    return SimpleNamespace(pitch=m.pitch_deg, cam_fwd=m.offset_fwd_m, height=height - m.offset_down_m,
                           max_range=rng, margin_px=20.0, speed=speed, accel=accel, fps=8.0)


def yaw_limits():
    """(rate rad/s, accel rad/s^2) the drone turns its nose with: config.yaml yaw: (sent to the v2 firmware)."""
    import yaml
    try:
        y = (yaml.safe_load((HERE / "miniproject" / "config.yaml").read_text()) or {}).get("yaw") or {}
    except OSError:
        y = {}
    return math.radians(float(y.get("rate_max_dps", 30))), math.radians(float(y.get("accel_max_dps2", 30)))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def fillet(corners, r_max: float):
    """Polyline through `corners` with each corner rounded by a circle arc (radius at most r_max,
    and small enough to fit the sides).  Returns dense points (0.01 m apart) and, per corner, the
    index where its arc starts."""
    C = [np.asarray(c, float) for c in corners]
    out, arc_start = [C[0]], []
    for k in range(1, len(C) - 1):
        a, v, b = C[k - 1], C[k], C[k + 1]
        d1, d2 = (v - a) / np.linalg.norm(v - a), (b - v) / np.linalg.norm(b - v)
        turn = math.atan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2))       # signed turn angle
        if abs(turn) < 1e-3:
            arc_start.append(len(out) - 1)
            continue
        tl = math.tan(abs(turn) / 2)
        r = min(r_max, 0.5 * np.linalg.norm(v - a) / tl, 0.5 * np.linalg.norm(b - v) / tl)
        t1, t2 = v - d1 * r * tl, v + d2 * r * tl
        n = np.array([-d1[1], d1[0]]) * (1 if turn > 0 else -1)                # towards the turn
        c = t1 + n * r
        seg = np.linspace(out[-1], t1, max(2, int(np.linalg.norm(t1 - out[-1]) / 0.01) + 1))[1:]
        out.extend(seg)
        arc_start.append(len(out) - 1)
        a0 = math.atan2(*(t1 - c)[::-1])
        th = a0 + turn * np.linspace(0, 1, max(8, int(abs(turn) * r / 0.01)))[1:]
        out.extend(np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)])
    seg = np.linspace(out[-1], C[-1], max(2, int(np.linalg.norm(C[-1] - out[-1]) / 0.01) + 1))[1:]
    out.extend(seg)
    return np.array(out), arc_start


def time_curved(dense, speed: float, accel: float, omega: float, end_hold: float = 0.0):
    """Rest-to-rest timing along one leg: cruise `speed`, at most `accel`, and slower where the
    path bends so the nose (following the path) never has to turn faster than 0.85 * omega."""
    seg = np.hypot(*np.diff(dense, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    hd = np.unwrap(np.arctan2(np.diff(dense[:, 1]), np.diff(dense[:, 0])))
    kap = np.zeros(len(dense))
    if len(hd) > 1:
        dk = np.abs(np.diff(hd)) / np.maximum(0.5 * (seg[:-1] + seg[1:]), 1e-6)
        kap[1:-1] = np.convolve(dk, np.ones(9) / 9, mode="same")                # light smoothing
    v = np.minimum(speed, 0.85 * omega / np.maximum(kap, 1e-9))
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):                                                   # speed up
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * seg[i - 1]))
    for i in range(len(v) - 2, -1, -1):                                          # slow down
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * seg[i]))
    vm = 0.5 * (v[1:] + v[:-1])
    dt = np.where(vm > 1e-6, seg / np.maximum(vm, 1e-6), 0.0)
    dt[(vm <= 1e-6) & (seg > 0)] = np.sqrt(2 * seg[(vm <= 1e-6) & (seg > 0)] / accel)
    tt = np.r_[0, np.cumsum(dt)]
    T = np.arange(0, tt[-1] + 1e-9, pc.DT)
    P = np.c_[np.interp(T, tt, dense[:, 0]), np.interp(T, tt, dense[:, 1])]
    P = np.vstack([P, dense[-1:]])
    if end_hold > 0:
        P = np.vstack([P, np.repeat(P[-1:], int(round(end_hold / pc.DT)), axis=0)])
    return P, np.interp(s, tt * 0 + s, tt)                                      # samples, time at each dense pt


def outward_path(L: float, W: float, half_len: float, half_wid: float, a):
    """Start -> straight ahead, then onto the near-RIGHT corner of a loop centred in the field -> once round it
    (inside on the drone's left, so 'nose 90 deg right of travel' = facing OUT) -> stop at the middle
    of the near side (nose already on the start) -> straight back to the start, nose on the start.
    Rounded corners, slower where it bends so the nose can keep up (config.yaml yaw limits).
    Returns the timed path and the [from, to) points where the nose is turned outward."""
    rate, _acc = yaw_limits()
    cx = L / 2
    x0, x1, y0, y1 = cx - half_len, cx + half_len, -half_wid, half_wid
    xs = max(0.5 * x0, x0 - 0.35)                    # straight ahead on the centre line until here
    legA, arcs = fillet([(0, 0), (xs, 0), (x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, 0)], R_CORNER)
    legB = np.linspace((x0, 0.0), (0.0, 0.0), max(2, int(x0 / 0.01) + 1))
    PA, _ = time_curved(legA, a.speed, a.accel, rate)
    PB, _ = time_curved(legB, a.speed, a.accel, rate, end_hold=END_HOLD_S)
    P = np.vstack([PA, PB[1:]])
    # outward from where the path turns onto the loop's right side until the stop at the near side
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(legA, axis=0).T))]
    cumP = np.r_[0, np.cumsum(np.hypot(*np.diff(PA, axis=0).T))]
    i_from = int(np.searchsorted(cumP, cum[arcs[1]]))
    return P, (i_from, len(PA) - 1)


def nose_headings(P, window, dt=pc.DT, rate=None, acc=None):
    """What the v2 yaw firmware flies (yaw-follow + outward window): target = direction of travel,
    + 90 deg inside the window, kept while not moving; the nose turns towards it with the turn rate
    ramped at <= acc, capped at rate, and slowed down in time to stop on the target."""
    if rate is None or acc is None:
        rate, acc = yaw_limits()
    yaw = np.zeros(len(P))
    h = w = want = 0.0
    for i in range(1, len(P)):
        dx, dy = P[i] - P[i - 1]
        if dx * dx + dy * dy > 1e-10:
            off = math.pi / 2 if window[0] <= i - 1 < window[1] else 0.0
            want = _wrap(math.atan2(dy, dx) + off)
        d = _wrap(want - h)
        vd = min(rate, math.sqrt(2 * acc * abs(d)))
        if vd * dt > abs(d):
            vd = abs(d) / dt
        vd = math.copysign(vd, d)
        w += max(-acc * dt, min(acc * dt, vd - w))
        h = _wrap(h + w * dt)
        yaw[i] = h
    return yaw


def coverage(P, yaw, a, L, W, cell=0.05, need=3):
    """Share of the field where a tag would be in the usable view in >= `need` frames (at a.fps)."""
    gx = np.arange(cell / 2, L, cell)
    gy = np.arange(-W / 2 + cell / 2, W / 2, cell)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    cnt = np.zeros(GX.shape, int)
    step = max(1, int(round(1.0 / (a.fps * pc.DT))))
    for i in range(0, len(P), step):
        x, y = P[i]
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        dx, dy = GX - x, GY - y
        cnt += pc.visible(dx * c + dy * s, -dx * s + dy * c, a)
    return gx, gy, cnt, float(np.mean(cnt >= need))


def plan_outward(L: float, W: float, a):
    """Smallest centred loop whose coverage is within 1 % of the best; returns (P, window, info)."""
    tried = []
    for d in np.arange(0.20, max(L, W) / 2 + 0.01, 0.05):              # inset of the loop from the field edge
        hl, hw = max(R_CORNER, L / 2 - d), max(R_CORNER, W / 2 - d)
        if tried and abs(tried[-1][1] - hl) < 1e-9 and abs(tried[-1][2] - hw) < 1e-9:
            continue
        P, win = outward_path(L, W, hl, hw, a)
        cov = coverage(P, nose_headings(P, win), a, L, W)[3]
        tried.append((d, hl, hw, cov, P, win))
    top = max(x[3] for x in tried)
    d, hl, hw, cov, P, win = max((x for x in tried if x[3] >= top - 0.01), key=lambda x: x[0])
    return P, win, dict(loop_length_m=round(2 * hl, 3), loop_width_m=round(2 * hw, 3),
                        coverage=round(cov, 3), best_coverage=round(top, 3), inset_m=round(d, 3))


def build(shape: str, length: float, width: float, speed: float = 0.2, accel: float = 0.12,
          lanes: int | None = None) -> np.ndarray:
    """Timed path, N x 2 (ahead, right) every 0.05 s."""
    L, W = float(length), float(width)
    if not (0.5 <= L <= 15 and 0.5 <= W <= 15):
        raise ValueError(f"length and width must be 0.5-15 m (got {L:g} x {W:g})")
    if not 0.05 <= speed <= MAX_SPEED:
        raise ValueError(f"--speed must be 0.05-{MAX_SPEED} m/s")
    a = SimpleNamespace(speed=speed, accel=accel, length=L, width=W)
    if shape == "ellipse":
        dense, stops = pc.plan_ellipse(a, L / 2, L / 2, W / 2, 1.0, 0.0)
    elif shape == "rectangle":
        r = min(0.25, W / 4, L / 4)
        dense, stops = pe.rectangle(L, W / 2, r)
    elif shape == "lawnmower_vertical":
        n = lanes or 4
        if n < 2:
            raise ValueError("--lanes must be 2 or more")
        sp = W / (n - 1)                                      # outer lanes on the box edges
        res = pc.plan_serpentine(a, False, sp, 0.0, sp / 2, L - sp / 2)
        if res is None:
            raise ValueError(f"{n} lanes need the box longer than {sp + 0.2:.2f} m; use more --lanes")
        dense, stops = res
    elif shape == "lawnmower_horizontal":
        n = lanes or 4
        sx = L / n                                            # lanes at L/n, 2L/n, ... L
        res = pc.plan_serpentine(a, True, sx, sx / 2, sx, L)
        if res is None:
            raise ValueError(f"{n} lanes leave no room across a {W:g} m box (U-turns {sx:.2f} m wide); "
                             f"use more --lanes")
        dense, stops = res
    else:
        raise ValueError(f"unknown shape {shape!r}; one of {', '.join(SHAPES)}")
    return pe.timed(dense, stops, a)


def check(P: np.ndarray, speed: float) -> None:
    v = np.hypot(*np.diff(P, axis=0).T) / pc.DT
    if v.max() > MAX_SPEED + 1e-6:
        raise ValueError(f"top speed {v.max():.2f} m/s is over the {MAX_SPEED} m/s cap")
    if len(P) > MAX_POINTS:
        T = len(P) * pc.DT
        path_s = T - (pe.CREEP_M / pe.CREEP_SPEED + 3.0)      # time moving at cruise
        need = speed * path_s / (MAX_POINTS * pc.DT - (T - path_s))
        hint = (f"--speed {math.ceil(need * 100) / 100:.2f}" if need <= MAX_SPEED
                else "fewer --lanes or a smaller box")
        raise ValueError(f"{T:.0f} s is longer than the drone takes in one go ({MAX_POINTS * pc.DT:.0f} s): "
                         f"use {hint}")


def name_for(shape: str, length: float, width: float, lanes: int | None) -> str:
    s = f"{shape}_{length:g}x{width:g}".replace(".", "p")
    return s + (f"_{lanes}lanes" if lanes and shape.startswith("lawnmower") else "")


def plot(P: np.ndarray, path: Path, shape: str, L: float, W: float, speed: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    import yaml
    tags = {}
    for line in (HERE / "tags_ref.csv").read_text().splitlines() if (HERE / "tags_ref.csv").exists() else []:
        s = line.split("#")[0].strip()
        try:
            i, n, e = s.split(",")[:3]
            tags[int(i)] = (float(n), float(e), "target")
        except ValueError:
            pass
    try:
        an = (yaml.safe_load((HERE / "miniproject" / "config.yaml").read_text()) or {}).get("anchors") or {}
        if an.get("home_tag") is not None:
            tags[int(an["home_tag"])] = (*map(float, an["home_tag_at"]), "home")
        if an.get("map_tag") is not None:
            tags[int(an["map_tag"])] = (*map(float, an["map_tag_at"]), "anchor")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    length_m = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
    fig, ax = plt.subplots(figsize=(5.2, 1.6 + 5.0 * max(L, 1.0) / max(W + 1.0, 1.0) * 0.55 + 2.4))
    ax.add_patch(Rectangle((-W / 2, 0), W, L, fill=False, ls="--", lw=1, ec="0.45"))
    ax.plot(P[:, 1], P[:, 0], "-", color="#2f6fb3", lw=1.8)
    k = len(P) // 6
    for i in range(k, len(P) - 1, k):                         # direction of travel
        d = P[min(i + 5, len(P) - 1)] - P[i]
        if np.hypot(*d) > 1e-4:
            ax.annotate("", xy=(P[i, 1] + d[1] * 3, P[i, 0] + d[0] * 3), xytext=(P[i, 1], P[i, 0]),
                        arrowprops=dict(arrowstyle="-|>", color="#2f6fb3", lw=1.2))
    ax.plot(0, 0, "o", color="k", ms=6)
    ax.annotate("start", (0, 0), textcoords="offset points", xytext=(7, -10), fontsize=8)
    for t, (tx, ty, kind) in tags.items():
        ax.plot(ty, tx, "s", ms=8, mfc="k" if kind != "target" else "white", mec="k")
        ax.annotate(f"{t}" + ("" if kind == "target" else f" ({kind})"), (ty, tx), textcoords="offset points",
                    xytext=(7, -3), fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=.25)
    ax.set_xlabel("right (m)")
    ax.set_ylabel("ahead of the start nose (m)")
    ax.set_title(f"{shape.replace('_', ' ')} {L:g} m x {W:g} m\n{length_m:.1f} m, {len(P) * pc.DT:.0f} s at "
                 f"{speed:g} m/s", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _tags():
    import yaml
    tags = {}
    f = HERE / "tags_ref.csv"
    for line in f.read_text().splitlines() if f.exists() else []:
        s = line.split("#")[0].strip()
        try:
            i, n, e = s.split(",")[:3]
            tags[int(i)] = (float(n), float(e), "target")
        except ValueError:
            pass
    try:
        an = (yaml.safe_load((HERE / "miniproject" / "config.yaml").read_text()) or {}).get("anchors") or {}
        if an.get("home_tag") is not None:
            tags[int(an["home_tag"])] = (*map(float, an["home_tag_at"]), "home")
        if an.get("map_tag") is not None:
            tags[int(an["map_tag"])] = (*map(float, an["map_tag_at"]), "anchor")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return tags


def plot_outward(P, Y, a, path: Path, L: float, W: float, speed: float, info: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
    except ImportError:
        return
    gx, gy, cnt, cv = coverage(P, Y, a, L, W)
    tags = {t: v for t, v in _tags().items() if 0 <= v[0] <= L and abs(v[1]) <= W / 2}
    step = max(1, int(round(1.0 / (a.fps * pc.DT))))
    seen = {}
    for t, (tx, ty, _k) in tags.items():
        n = 0
        for i in range(0, len(P), step):
            c, s = math.cos(Y[i]), math.sin(Y[i])
            dx, dy = tx - P[i, 0], ty - P[i, 1]
            n += bool(pc.visible(np.array(dx * c + dy * s), np.array(-dx * s + dy * c), a))
        seen[t] = n >= 3
    fig, ax = plt.subplots(figsize=(5.6, 2.2 + 5.2 * L / max(W, 1.0) * 0.62))
    ax.pcolormesh(gy, gx, np.ma.masked_less((cnt >= 3).astype(float), 0.5), cmap=ListedColormap(["#cfe3c7"]),
                  shading="nearest", zorder=0)
    ax.add_patch(Rectangle((-W / 2, 0), W, L, fill=False, lw=1.3, ec="#1f2328", zorder=3))
    ax.plot(P[:, 1], P[:, 0], "-", color="#2f6fb3", lw=1.8, zorder=4)
    idx = np.arange(0, len(P), int(1.5 / pc.DT))
    ax.quiver(P[idx, 1], P[idx, 0], np.sin(Y[idx]), np.cos(Y[idx]), angles="xy", scale_units="xy",
              scale=4.5, width=0.006, color="#1f2328", alpha=.75, zorder=5)
    ax.plot(0, 0, "o", color="#1f2328", ms=6, zorder=6)
    for t, (tx, ty, kind) in tags.items():
        ok = seen[t]
        ax.plot(ty, tx, "s", ms=9, mfc="#1f2328" if ok else "white", mec="#1f2328", mew=1.3, zorder=7)
        lab = f"{t}" + ("" if kind == "target" else f" ({kind})") + ("" if ok else " not seen")
        ax.annotate(lab, (ty, tx), textcoords="offset points", xytext=(7, -3), fontsize=8,
                    color="#1f2328" if ok else "#b42318", zorder=8)
    ax.set_aspect("equal")
    ax.set_xlim(-W / 2 - 0.3, W / 2 + 0.3)
    ax.set_ylim(-0.3, L + 0.3)
    ax.grid(alpha=.2)
    ax.set_xlabel("right (m)")
    ax.set_ylabel("ahead of the start nose (m)")
    length_m = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
    ax.set_title(f"outward rectangle, field {L:g} x {W:g} m: loop {info['loop_length_m']:.2f} x "
                 f"{info['loop_width_m']:.2f} m\n{length_m:.1f} m, {len(P) * pc.DT:.0f} s at {speed:g} m/s; "
                 f"camera covers {100 * cv:.0f}% (green); arrows = nose", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make(shape: str, length: float, width: float, out_dir: Path | str = HERE / "trajectories",
         speed: float = 0.2, accel: float = 0.12, lanes: int | None = None, out: str | None = None,
         picture: bool = True, height: float = 0.8) -> Path:
    """Build, check and write the CSV (+ a .png beside it).  Returns the CSV path."""
    path = Path(out) if out else Path(out_dir) / f"{name_for(shape, length, width, lanes)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if shape == "outward_rectangle":
        import json
        if not (1.0 <= length <= 15 and 1.0 <= width <= 15):
            raise ValueError(f"field must be 1-15 m each way (got {length:g} x {width:g})")
        a = camera_model(height, speed, accel)
        P, win, info = plan_outward(length, width, a)
        check(P, speed)
        Y = nose_headings(P, win)
        pc.write_csv(path, P, Y)
        side = {"shape": shape, "field_m": [length, width], "height_m": height, **info,
                "nose_outward_from": win[0], "nose_outward_to": win[1], "land_at_end": True,
                "firmware": "yaw v2 patch (CMD_TRAJ_START flags 0x03 + window + yaw limits)"}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1) + "\n")
        if picture:
            plot_outward(P, Y, a, path.with_suffix(".png"), length, width, speed, info)
        return path
    P = build(shape, length, width, speed, accel, lanes)
    check(P, speed)
    pc.write_csv(path, P, np.zeros(len(P)))
    if picture:
        plot(P, path.with_suffix(".png"), shape, length, width, speed)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shape", choices=SHAPES)
    ap.add_argument("length", type=float, help="metres straight ahead of the start nose")
    ap.add_argument("width", type=float, help="metres across, centred on the start circle")
    ap.add_argument("--speed", type=float, default=0.2, help="cruise speed, m/s (default 0.2, max 0.3)")
    ap.add_argument("--accel", type=float, default=0.12, help="speed-up / slow-down, m/s^2 (default 0.12)")
    ap.add_argument("--lanes", type=int, default=None, help="lawnmowers: number of lanes (default 4)")
    ap.add_argument("--out", default=None, help="CSV to write (default trajectories\\<shape>_<L>x<W>.csv)")
    ap.add_argument("--no-plot", action="store_true", help="do not save the .png picture")
    ap.add_argument("--height", type=float, default=0.8,
                    help="outward_rectangle: flight height for the camera model (m, default 0.8)")
    a = ap.parse_args()
    try:
        path = make(a.shape, a.length, a.width, speed=a.speed, accel=a.accel, lanes=a.lanes, out=a.out,
                    picture=not a.no_plot, height=a.height)
    except ValueError as e:
        print(f"not made: {e}")
        return 1
    P = np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:3]
    length_m = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
    print(f"{path}: {a.shape} {a.length:g} m x {a.width:g} m, {length_m:.1f} m, {len(P) * pc.DT:.0f} s at "
          f"{a.speed:g} m/s, {len(P)} points" + ("" if a.no_plot else f"; picture {path.with_suffix('.png').name}"))
    print(f"fly it:  python run_baseline.py ... --trajectory {path.relative_to(HERE) if path.is_relative_to(HERE) else path}"
          f"   (or straight away: --trajectory {a.shape} {a.length:g} {a.width:g})")
    return 0


# ------------------------------------------------------------------ slow turns (any firmware)
def _bezier(p0, h0, p1, h1, max_step_deg=0.5):
    """Smooth curve from p0 (leaving along heading h0) to p1 (arriving along h1), many points."""
    L = np.hypot(*(p1 - p0))
    c0 = p0 + 0.42 * L * np.array([math.cos(h0), math.sin(h0)])
    c1 = p1 - 0.42 * L * np.array([math.cos(h1), math.sin(h1)])
    turn = abs(_wrap(h1 - h0))
    n = max(20, int(math.ceil(math.degrees(turn) / max_step_deg)) * 3, int(3 * L / 0.004))
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c0 + 3 * (1 - u) * u ** 2 * c1 + u ** 3 * p1


def slow_turns(P, turn_dps: float = 20.0, speed: float = 0.2, accel: float = 0.12, r: float = 0.06,
               end_hold: float = 2.0, start_heading: float = 0.0, sharp_deg: float = 3.0):
    """Re-time a path so the direction of travel (= the nose with --yaw-follow) turns no faster than
    turn_dps, on ANY firmware.  Sharp corners (incl. the first turn away from the take-off heading
    and the turn into the final creep) become short smooth curves drawn with many points; the drone
    slows down wherever the path bends so it turns at <= turn_dps; straights stay at `speed`.
    The route is otherwise unchanged (corners cut by a few cm)."""
    om, ds = math.radians(turn_dps), 0.005
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]
    G0 = P[keep]
    s0 = np.r_[0, np.cumsum(np.hypot(*np.diff(G0, axis=0).T))]
    S = np.r_[np.arange(0, s0[-1], ds), s0[-1]]
    G = np.c_[np.interp(S, s0, G0[:, 0]), np.interp(S, s0, G0[:, 1])]
    h = np.arctan2(np.diff(G[:, 1]), np.diff(G[:, 0]))                       # heading of segment j
    dh = np.r_[_wrap(h[0] - start_heading), [_wrap(h[j] - h[j - 1]) for j in range(1, len(h))]]
    sharp = np.where(np.abs(dh) > math.radians(sharp_deg))[0]               # vertex j = start of segment j
    clusters = []
    for j in sharp:
        if clusters and j - clusters[-1][1] <= int(0.03 / ds):
            clusters[-1][1] = j
        else:
            clusters.append([j, j])
    pieces, pos, tail = [], 0, None
    for ja, jb in clusters:
        tot = float(np.sum(dh[ja:jb + 1]))
        if abs(tot) > math.radians(100):
            # big turn (e.g. arriving back at the start, then the creep ahead): a turnaround arc of
            # radius r_turn from just before the corner, then on to the path a little further on
            r_t = 0.08
            ia = max(pos, ja - int(0.02 / ds)) if ja > 0 else 0
            h_in = start_heading if ia == 0 else h[ia - 1]
            n = np.array([-math.sin(h_in), math.cos(h_in)]) * (1 if tot > 0 else -1)
            c = G[ia] + n * r_t
            a0 = math.atan2(*(G[ia] - c)[::-1])
            th = a0 + tot * np.linspace(0, 1, max(20, int(abs(math.degrees(tot)) / 0.5)))
            arc = np.c_[c[0] + r_t * np.cos(th), c[1] + r_t * np.sin(th)]
            pieces.append(G[pos:ia + 1])
            ib = jb + int(0.35 / ds)
            if ib >= len(G) - 1:                         # near the end: finish with the turn + 5 cm
                h_e = h_in + tot
                tail = np.vstack([arc[1:], arc[-1] + np.outer(np.linspace(0.01, 0.05, 5),
                                                              [math.cos(h_e), math.sin(h_e)])])
                pos = None
                break
            pieces.append(arc[1:])
            pieces.append(_bezier(arc[-1], h_in + tot, G[ib], h[ib])[1:-1])
            pos = ib
            continue
        k = int(math.ceil(r * math.tan(abs(tot) / 2) / ds)) + 1
        ia = max(pos, ja - k) if ja > 0 else 0
        ib = min(len(G) - 1, jb + k)
        h_in = start_heading if ia == 0 else h[ia - 1]
        h_out = h[ib] if ib < len(h) else h[-1]
        pieces.append(G[pos:ia + 1])
        pieces.append(_bezier(G[ia], h_in, G[ib], h_out)[1:-1])
        pos = ib
    pieces.append(tail if pos is None else G[pos:])
    D = np.vstack(pieces)
    D = D[np.r_[True, np.hypot(*np.diff(D, axis=0).T) > 1e-7]]
    seg = np.hypot(*np.diff(D, axis=0).T)
    hd = np.arctan2(np.diff(D[:, 1]), np.diff(D[:, 0]))
    tr = np.abs(np.r_[_wrap(hd[0] - start_heading), [_wrap(hd[j] - hd[j - 1]) for j in range(1, len(hd))]])
    kap = tr / np.maximum(seg, 1e-6)                                          # turn per metre, segment j
    kap = np.maximum.reduce([np.roll(kap, s) for s in range(-4, 5)])          # look a few points around
    v_seg = np.minimum(speed, om / np.maximum(kap, 1e-9))
    v = np.r_[v_seg[0], np.minimum(v_seg[1:], v_seg[:-1]), v_seg[-1]]
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * accel * seg[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * accel * seg[i]))
    vm = 0.5 * (v[1:] + v[:-1])
    dt = np.where(vm > 1e-3, seg / np.maximum(vm, 1e-3), np.sqrt(2 * seg / accel))
    tt = np.r_[0, np.cumsum(dt)]
    T = np.r_[np.arange(0, tt[-1], pc.DT), tt[-1]]
    Q = np.c_[np.interp(T, tt, D[:, 0]), np.interp(T, tt, D[:, 1])]
    return np.vstack([Q, np.repeat(Q[-1:], int(round(end_hold / pc.DT)), axis=0)])


def firmware_nose(P, start=0.0, step=0.1):
    """The nose on the FIRST yaw-follow firmware: aims along the path, moves <= 0.1 rad per 50 ms."""
    h, want, out = start, start, [start]
    for i in range(1, len(P)):
        dx, dy = P[i] - P[i - 1]
        if dx * dx + dy * dy > 1e-8:
            want = math.atan2(dy, dx)
        h = _wrap(h + max(-step, min(step, _wrap(want - h))))
        out.append(h)
    return np.array(out)


if __name__ == "__main__":
    raise SystemExit(main())
