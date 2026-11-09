"""Trajectory experiment: 4 path shapes x 2 yaw modes = 8 trajectories in the SAME box.

  python plan_experiment.py                      -> experiment_plans\\E1_... E8_... .csv + overview

Controlled (identical for every trajectory):
  box        x = 0 .. LENGTH m ahead of the start nose, y = -WIDTH/2 .. +WIDTH/2 (right = +);
             every path touches all four edges of the box and nothing outside it
  speed      cruise --speed m/s, --accel m/s^2 (smooth U-turns are flown at cruise speed)
  height     z = 0 in the CSV = the 0.5 m take-off hover; fly higher with run_baseline --height
             (2 Nov: --height 0.8 for every flight).  --height here only sets the camera model.
  start/end  starts on the start circle; ends with a slow 0.15 m creep straight ahead, so the
             nose faces tag 20 again for the landing (the yaw-follow firmware keeps the LAST
             heading after a path ends).  The constant-yaw paths creep too, so each pair is the
             same path.

Independent variables: SHAPE and YAW
  E1 ellipse                   one lap of the ellipse that fills the box        yaw constant
  E2 lawnmower_vertical        4 lanes running ahead/back, 0.5 m apart, U-turns  yaw constant
  E3 lawnmower_horizontal      4 lanes running left/right, stepping ahead        yaw constant
  E4 ellipse_yaw               = E1, nose turning along the path                 --yaw-follow
  E5 lawnmower_vertical_yaw    = E2, nose turning along the path                 --yaw-follow
  E6 lawnmower_horizontal_yaw  = E3, nose turning along the path                 --yaw-follow
  E7 rectangle                 the box outline, corners rounded (0.25 m)        yaw constant
  E8 rectangle_yaw             = E7, nose turning along the path                 --yaw-follow
  (E7/E8 were added after E1-E6, so the numbers of E1-E6 stay as they were.)
  Rectangle corners are rounded so the drone keeps cruising and the nose, turning at most
  1 rad/s, keeps up with a 90 deg corner (0.2 m/s on a 0.25 m radius = 0.8 rad/s).

The yaw versions only turn on a drone flashed with the yaw-follow firmware patch (old firmware
flies them nose-forward).  Their yaw column is what that firmware does: heading along the path,
turning at most 1 rad/s, kept while hovering.  The firmware works it out itself from the path;
the column is there for the camera model below and as a record.

Writes experiment_plans/<name>.csv (t,x,y,z,yaw_rad every 0.05 s), overview.png and summary.txt
(length, time, and which tags the camera model expects to see, with the mount the flights use:
mapper_config_current.yaml from set_camera_mount.py, else the 29 Oct one).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import plan_coverage as pc

HERE = Path(__file__).resolve().parent
CREEP_M, CREEP_SPEED = 0.15, 0.04          # final creep straight ahead (turns the nose forward)
YAW_RATE = 1.0                              # rad/s, the yaw-follow firmware's turn-rate limit


def shapes(a):
    """(name, dense path, stop indices) for the three shapes; all start and end at (0, 0)."""
    L, W = a.length, a.width
    out = {}
    out["ellipse"] = pc.plan_ellipse(a, L / 2, L / 2, W / 2, 1.0, 0.0)
    # lanes ahead/back at y = -0.75, -0.25, +0.25, +0.75; U-turns of radius 0.25 reach x = 0 and x = L
    sp = W / 3
    out["lawnmower_vertical"] = pc.plan_serpentine(a, False, sp, 0.0, sp / 2, L - sp / 2)
    # lanes left/right at 4 steps ahead (the last at x = L); U-turns reach y = +-W/2
    sx = L / 4
    out["lawnmower_horizontal"] = pc.plan_serpentine(a, True, sx, sx / 2, sx, L)
    out["rectangle"] = rectangle(L, W / 2, 0.25)
    return out


def rectangle(L: float, h: float, r: float):
    """Outline of the box, clockwise from the start (0, 0) on the near edge: right, up the right
    edge, across the far edge, down the left edge, back to the start.  Corners are quarter circles
    of radius r, flown at cruise speed (no stops)."""
    def line(p0, p1):
        n = max(2, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.01) + 1)
        return np.linspace(p0, p1, n)

    def arc(cx, cy, a0, a1):
        th = np.linspace(a0, a1, 40)
        return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
    parts = [line((0.0, 0.0), (0.0, h - r)), arc(r, h - r, math.pi, math.pi / 2),
             line((r, h), (L - r, h)), arc(L - r, h - r, math.pi / 2, 0.0),
             line((L, h - r), (L, -h + r)), arc(L - r, -h + r, 0.0, -math.pi / 2),
             line((L - r, -h), (r, -h)), arc(r, -h + r, -math.pi / 2, -math.pi),
             line((0.0, -h + r), (0.0, 0.0))]
    dense = np.vstack([parts[0]] + [q[1:] for q in parts[1:]])
    return dense, [0, len(dense) - 1]


def timed(dense, stops, a):
    """Cruise along the path, pause 1 s on the start circle, creep CREEP_M ahead slowly, hold 2 s."""
    P1 = pc.time_path(dense, stops, a, end_hold=1.0)
    slow = SimpleNamespace(**{**vars(a), "speed": CREEP_SPEED})
    creep = np.linspace(P1[-1], P1[-1] + [CREEP_M, 0.0], 60)
    P2 = pc.time_path(creep, [0, len(creep) - 1], slow, end_hold=2.0)
    return np.vstack([P1, P2[1:]])


def follow_yaw(P, dt):
    """Heading the yaw-follow firmware flies: along the path, <= YAW_RATE, kept when not moving."""
    yaw = np.zeros(len(P))
    h = 0.0
    lim = YAW_RATE * dt
    for i in range(1, len(P)):
        dx, dy = P[i] - P[i - 1]
        if dx * dx + dy * dy > 1e-8:
            d = (math.atan2(dy, dx) - h + math.pi) % (2 * math.pi) - math.pi
            h = (h + max(-lim, min(lim, d)) + math.pi) % (2 * math.pi) - math.pi
        yaw[i] = h
    return yaw


def tag_frames(P, yaw, a, tags):
    """Frames (at a.fps) in which each tag is inside the camera's usable view."""
    step = max(1, int(round(1.0 / (a.fps * pc.DT))))
    cnt = {t: 0 for t in tags}
    for i in range(0, len(P), step):
        x, y = P[i]
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        for t, (tx, ty) in tags.items():
            dx, dy = tx - x, ty - y
            if pc.visible(np.array(dx * c + dy * s), np.array(-dx * s + dy * c), a):
                cnt[t] += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=float, default=2.3, help="box length straight ahead (m)")
    ap.add_argument("--width", type=float, default=1.5, help="box width, centred on the start (m)")
    ap.add_argument("--speed", type=float, default=0.20, help="cruise speed (m/s)")
    ap.add_argument("--accel", type=float, default=0.12, help="speed-up / slow-down (m/s^2)")
    ap.add_argument("--height", type=float, default=0.5,
                    help="flight height for the camera model (m); the CSVs stay at z = 0, fly with "
                         "run_baseline --height.  Default 0.5")
    ap.add_argument("--max-range", type=float, default=None,
                    help="tag range limit (m); default = config.yaml gate.max_range_m + (height - 0.5)/3, "
                         "the same rule run_baseline uses")
    ap.add_argument("--refs", default=str(HERE / "tags_ref.csv"))
    ap.add_argument("--out", default=str(HERE / "experiment_plans"))
    a = ap.parse_args()
    from miniproject.settings import mount_from_mapper
    opt = HERE / "optimization_comparison" / "laptop"
    mfile = next(p for p in (opt / "mapper_config_current.yaml", opt / "mapper_config_29_Oct.yaml") if p.exists())
    m = mount_from_mapper(mfile)                       # the same mount the flights use
    rng = a.max_range
    if rng is None:
        from miniproject.settings import load_settings
        rng = load_settings(str(HERE / "miniproject" / "config.yaml")).gate.max_range_m
        rng += max(0.0, a.height - 0.5) / 3.0
    cam = SimpleNamespace(pitch=m.pitch_deg, cam_fwd=m.offset_fwd_m, height=a.height - m.offset_down_m,
                          max_range=rng, margin_px=20.0,
                          speed=a.speed, accel=a.accel, fps=8.0, length=a.length, width=a.width)
    tags = {}
    for line in Path(a.refs).read_text().splitlines():
        s = line.split("#")[0].strip()
        if s:
            try:
                i, n, e = s.split(",")[:3]
                tags[int(i)] = (float(n), float(e))
            except ValueError:
                pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("E[0-9]_*.csv"):                  # earlier experiment sets
        old.unlink()
    rows = []
    base = shapes(cam)
    number = {("ellipse", False): 1, ("lawnmower_vertical", False): 2, ("lawnmower_horizontal", False): 3,
              ("ellipse", True): 4, ("lawnmower_vertical", True): 5, ("lawnmower_horizontal", True): 6,
              ("rectangle", False): 7, ("rectangle", True): 8}       # E1-E6 kept their numbers
    for with_yaw in (False, True):
        for shape, (dense, stops) in base.items():
            name = f"E{number[(shape, with_yaw)]}_{shape}" + ("_yaw" if with_yaw else "")
            P = timed(dense, stops, cam)
            yaw = follow_yaw(P, pc.DT) if with_yaw else np.zeros(len(P))
            pc.write_csv(out / f"{name}.csv", P, yaw)
            L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
            v = np.hypot(*np.diff(P, axis=0).T) / pc.DT
            frames = tag_frames(P, yaw, cam, tags)
            rows.append(dict(name=name, shape=shape, yaw=with_yaw, P=P, Y=yaw, length=L, time=len(P) * pc.DT,
                             vmax=float(v.max()), x=(P[:, 0].min(), P[:, 0].max()),
                             y=(P[:, 1].min(), P[:, 1].max()), frames=frames,
                             seen=[t for t, c in frames.items() if c >= 3],
                             end_heading=math.degrees(yaw[-1])))
    lines = [f"Trajectory experiment: box {a.length:.2f} m ahead x {a.width:.2f} m wide, cruise {a.speed} m/s, "
             f"accel {a.accel} m/s^2, height {a.height:.2f} m; ends with a {CREEP_M} m creep ahead at {CREEP_SPEED} m/s",
             "yaw versions: fly with --yaw-follow on the yaw-follow firmware (nose along the path, <= 1 rad/s)",
             f"camera model ({mfile.name}): {m.pitch_deg:.0f} deg below horizontal, {m.offset_fwd_m * 100:.1f} cm "
             f"ahead, range limit {rng:.2f} m, 8 fps; tag 'expected' = inside the usable view in >= 3 frames", ""]
    lines.append(f"{'trajectory':30s} {'length':>7s} {'time':>6s} {'vmax':>5s}  {'x range':>13s}  {'y range':>14s}  "
                 f"{'end nose':>8s}  expected tags (frames in view)")
    for r in rows:
        fr = ", ".join(f"{t}:{c}" for t, c in r["frames"].items())
        lines.append(f"{r['name']:30s} {r['length']:6.2f}m {r['time']:5.1f}s {r['vmax']:4.2f}  "
                     f"{r['x'][0]:+.2f}..{r['x'][1]:+.2f}  {r['y'][0]:+.2f}..{r['y'][1]:+.2f}   "
                     f"{r['end_heading']:+6.1f}d   {len(r['seen'])}/{len(tags)}  ({fr})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        ncol = len(base)
        fig, axs = plt.subplots(2, ncol, figsize=(3.3 * ncol, 11.6), sharex=True, sharey=True)
        for ax, r in zip(axs.flat, rows):
            P, Y = r["P"], r["Y"]
            ax.add_patch(Rectangle((-a.width / 2, 0), a.width, a.length, fill=False, ls="--", lw=1, ec="0.5"))
            ax.plot(P[:, 1], P[:, 0], "-", color="#f28e2b" if r["yaw"] else "#4e79a7", lw=1.6)
            idx = np.arange(0, len(P), int(2.0 / pc.DT))          # nose direction every 2 s
            ax.quiver(P[idx, 1], P[idx, 0], np.sin(Y[idx]), np.cos(Y[idx]), angles="xy", scale_units="xy",
                      scale=5, width=0.006, color="k", alpha=.55)
            ax.plot(0, 0, "o", color="k", ms=6)
            for t, (tx, ty) in tags.items():
                ok = t in r["seen"]
                ax.plot(ty, tx, "s", ms=9, mfc="#59a14f" if ok else "white", mec="#59a14f" if ok else "#e15759")
                ax.annotate(str(t), (ty, tx), textcoords="offset points", xytext=(7, -3), fontsize=8)
            ax.set_title(f"{r['name'].split('_', 1)[0]} {r['shape'].replace('_', ' ')}"
                         f"{' + yaw' if r['yaw'] else ''}\n{r['length']:.1f} m, {r['time']:.0f} s, "
                         f"{len(r['seen'])}/{len(tags)} tags expected", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-0.3, 3.1)
            ax.grid(alpha=.25)
        for ax in axs[-1]:
            ax.set_xlabel("right (m)")
        for ax in axs[:, 0]:
            ax.set_ylabel("ahead of the start nose (m)")
        fig.suptitle(f"Same box and speed; flown at {a.height:.1f} m, camera {cam.pitch:.0f} deg below horizontal, "
                     f"tag range limit {rng:.1f} m. Top: yaw constant. Bottom: nose turns along the path.\n"
                     "Arrows = nose direction every 2 s; green square = tag expected in view, hollow red = not",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
        fig.savefig(out / "overview.png", dpi=120)
        print(f"\nwrote {len(rows)} CSVs, summary.txt and overview.png in {out}")
    except ImportError:
        print(f"\nwrote {len(rows)} CSVs and summary.txt in {out} (matplotlib missing: no overview.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
