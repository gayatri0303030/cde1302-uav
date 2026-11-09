"""Re-localise every tag of a finished run with several methods and score them against tape truth.

  python evaluate_tags.py logs\\fly_<time> --refs tags_ref.csv
  python evaluate_tags.py logs\\fly_<time>                       (no truth: shows spread only)

Uses only files the run already wrote (raw_detections.csv + events.jsonl), so it works on old
runs too and never touches the drone.  Each METHOD turns the raw AprilTag corners of every accepted
detection into a floor position; each FUSION turns all detections of a tag into one answer:

  methods
    firmware-pnp      tag pose from the drone's firmware calibration, no lens distortion
                      (exactly what run_baseline reported live)
    checker-pnp       same, but with the checkerboard calibration (K + 5 distortion terms)
    checker-ray       checkerboard calibration; the 4 corner rays are intersected with the floor at
                      the camera height (PX4 z if logged, else --height, else the run's median
                      checker-pnp height).  Does not depend on the printed tag size.
  fusions
    inlier-wmean      median -> keep within 0.20 m -> 1/range^2 weighted mean (the live rule)
    near-only         as above but only detections closer than --near m (horizontal)

--refs: "id,north_m,east_m" measured with a tape from the drone's start mark, +north = the nose
direction at start (the same frame as the live TAG_REPORT).  Writes evaluation.csv and
evaluation.png into the run folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

FIRMWARE_K = np.array([[163.5047216, 0, 154.00573087], [0, 153.22210511, 107.10222796], [0, 0, 1]])
CHECKER_K = np.array([[183.29255671332464, 0, 161.0350214076591],
                      [0, 182.85805282226417, 120.12225139955437], [0, 0, 1]])
CHECKER_D = np.array([-0.09528473439616442, 0.10853822148508256, -0.0003212610220958575,
                      0.002416692169986992, -0.045389390933938116])
APRILTAG_TO_OPENCV_PX = 0.5      # AprilTag corners use pixel-edge coordinates, OpenCV pixel centres


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_origin(run: Path):
    """(x0, y0, yaw0) of the start frame in PX4 local coordinates, from events.jsonl."""
    ev = run / "events.jsonl"
    if ev.exists():
        for line in ev.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") != "ORIGIN_SET":
                continue
            if e.get("origin_x") is not None:
                return float(e["origin_x"]), float(e["origin_y"]), float(e["origin_yaw_rad"]), "exact"
            m = re.search(r"\(([-+\d.]+),\s*([-+\d.]+)\).*heading\s*([-+\d.]+)\s*deg", e.get("text", ""))
            if m:
                return (float(m.group(1)), float(m.group(2)), math.radians(float(m.group(3))),
                        "from log text (2 dp, 1 deg)")
    return None


def to_start(n, e, origin):
    if origin is None:
        return n, e
    x0, y0, yaw0 = origin[:3]
    dn, de = n - x0, e - y0
    c, s = math.cos(yaw0), math.sin(yaw0)
    return dn * c + de * s, -dn * s + de * c


def cam_to_body(p_cam, pitch_deg, off_fwd):
    """OpenCV optical (right, down, forward) -> body (forward, right, down)."""
    a = math.radians(pitch_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = p_cam
    return np.array([off_fwd + c * z - s * y, x, s * z + c * y])


def body_to_local(b, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + b[0] * c - b[1] * s, y + b[0] * s + b[1] * c


def corners_of(r):
    return np.array([[float(r[f"corner_{i}_u"]), float(r[f"corner_{i}_v"])] for i in range(4)])


def pnp(corners, K, D, size, shift):
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    ok, rv, tv = cv2.solvePnP(obj, corners - shift, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rv)
    t = tv.ravel()
    return t, abs(R[:, 2] @ t)          # translation, camera height above the tag plane


def ray_floor(corners, K, D, shift, cam_height, pitch_deg):
    """Centre of the 4 corner rays intersected with the floor, in body (fwd, right) about the camera."""
    nrm = cv2.undistortPoints((corners - shift).reshape(-1, 1, 2), K, D).reshape(4, 2)
    pts = []
    for u, v in nrm:
        d = cam_to_body((u, v, 1.0), pitch_deg, 0.0)
        if d[2] <= 0.05 * np.linalg.norm(d):
            return None
        k = cam_height / d[2]
        pts.append(d * k)
    return np.mean(pts, axis=0)


def fuse(obs, radius=0.20, near=None):
    if near is not None:
        obs = [o for o in obs if o[2] <= near] or []
    if not obs:
        return None
    a = np.array(obs)
    mn, me = np.median(a[:, 0]), np.median(a[:, 1])
    inl = a[np.hypot(a[:, 0] - mn, a[:, 1] - me) <= radius]
    if len(inl) == 0:
        inl = a
    w = 1.0 / np.maximum(inl[:, 2], 0.1) ** 2
    n, e = (w @ inl[:, 0]) / w.sum(), (w @ inl[:, 1]) / w.sum()
    return n, e, len(a), float(np.hypot(inl[:, 0].std(), inl[:, 1].std()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="logs\\fly_<time> or logs\\observe_<time> folder")
    ap.add_argument("--refs", help="truth: id,north_m,east_m in the start frame")
    ap.add_argument("--tag-size", type=float, default=0.12, help="black square side (m)")
    ap.add_argument("--pitch", type=float, default=45.0, help="camera pitch, nose-down (deg)")
    ap.add_argument("--offset-fwd", type=float, default=0.02, help="camera ahead of body origin (m)")
    ap.add_argument("--height", type=float, help="camera height above floor for checker-ray (m)")
    ap.add_argument("--near", type=float, default=0.6, help="near-only fusion: max horizontal range (m)")
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--min-side-px", type=float, default=0,
                    help="drop detections whose tag is smaller than this in the image (px)")
    ap.add_argument("--edge-px", type=float, default=0,
                    help="drop detections whose tag centre is this close to the image border (px)")
    ap.add_argument("--ids", help="only these tag ids, e.g. 0,1,2,3 (drops false IDs)")
    a = ap.parse_args()

    run = Path(a.run)
    with open(run / "raw_detections.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("accepted") == "1"]
    def keep(r):
        c = corners_of(r)
        side = float(np.mean(np.hypot(*(c - np.roll(c, 1, axis=0)).T)))
        cu, cv = c.mean(axis=0)
        if a.ids and int(r["tag_id"]) not in {int(i) for i in a.ids.split(",")}:
            return False
        if side < a.min_side_px:
            return False
        return a.edge_px <= cu <= 320 - a.edge_px and a.edge_px <= cv <= 240 - a.edge_px
    n_all = len(rows)
    rows = [r for r in rows if keep(r)]
    if len(rows) != n_all:
        print(f"filters kept {len(rows)} of {n_all} detections "
              f"(min side {a.min_side_px:g} px, edge {a.edge_px:g} px, ids {a.ids or 'all'})")
    origin = load_origin(run)
    print(f"{len(rows)} accepted detections in {run}")
    print("start frame origin:", "NONE (reporting PX4 local frame)" if origin is None else
          f"PX4 ({origin[0]:+.3f}, {origin[1]:+.3f}) yaw {math.degrees(origin[2]):+.1f} deg [{origin[3]}]")

    refs = {}
    if a.refs:
        for ln in Path(a.refs).read_text().splitlines():
            p = [x.strip() for x in ln.split("#")[0].split(",")]
            if len(p) >= 3 and p[0].lstrip("-").isdigit():
                refs[int(p[0])] = (float(p[1]), float(p[2]))

    # camera height for the ray method
    chk_heights = [h for r in rows if (res := pnp(corners_of(r), CHECKER_K, CHECKER_D, a.tag_size,
                                                  APRILTAG_TO_OPENCV_PX)) for h in [res[1]]]
    run_height = float(np.median(chk_heights)) if chk_heights else 0.5
    if a.height:
        ray_h_src = f"--height {a.height:.3f} m"
    else:
        ray_h_src = f"median checker-pnp camera height {run_height:.3f} m (PX4 z used where logged)"
    print("checker-ray height:", ray_h_src)

    methods = {"firmware-pnp": {}, "checker-pnp": {}, "checker-ray": {}}
    for r in rows:
        x, y, yaw = f(r["px4_x_m"]), f(r["px4_y_m"]), f(r["px4_yaw_rad"])
        if None in (x, y, yaw):
            continue
        tid = int(r["tag_id"])
        c = corners_of(r)
        # firmware-pnp: re-solve with the firmware K (same as the logged tag_t*, but uses --tag-size)
        for name, K, D, sh in (("firmware-pnp", FIRMWARE_K, np.zeros(5), 0.0),
                               ("checker-pnp", CHECKER_K, CHECKER_D, APRILTAG_TO_OPENCV_PX)):
            res = pnp(c, K, D, a.tag_size, sh)
            if res is None:
                continue
            b = cam_to_body(res[0], a.pitch, a.offset_fwd)
            n, e = body_to_local(b, x, y, yaw)
            methods[name].setdefault(tid, []).append((*to_start(n, e, origin), math.hypot(b[0], b[1])))
        pz = f(r.get("px4_z_m"))
        h = a.height or (-pz if pz is not None and pz < -0.1 else run_height)
        b = ray_floor(c, CHECKER_K, CHECKER_D, APRILTAG_TO_OPENCV_PX, h, a.pitch)
        if b is not None:
            b[0] += a.offset_fwd
            n, e = body_to_local(b, x, y, yaw)
            methods["checker-ray"].setdefault(tid, []).append((*to_start(n, e, origin), math.hypot(b[0], b[1])))

    out_rows, summary = [], []
    for mname, per in methods.items():
        for fname, near in (("inlier-wmean", None), (f"near<{a.near:g}m", a.near)):
            errs = []
            for tid in sorted(per):
                fu = fuse(per[tid], near=near)
                if fu is None or fu[2] < a.min_obs:
                    continue
                n, e, cnt, spread = fu
                err = math.hypot(n - refs[tid][0], e - refs[tid][1]) if tid in refs else None
                if err is not None:
                    errs.append(err)
                out_rows.append({"method": mname, "fusion": fname, "tag_id": tid, "north_m": round(n, 4),
                                 "east_m": round(e, 4), "n_obs": cnt, "spread_m": round(spread, 4),
                                 "error_m": "" if err is None else round(err, 4)})
            summary.append((mname, fname, errs))

    print()
    tags = sorted({r["tag_id"] for r in out_rows})
    hdr = f"{'method':14s} {'fusion':13s} " + " ".join(f"{'tag ' + str(t):>18s}" for t in tags)
    print(hdr + ("   RMSE   max" if refs else ""))
    for mname, fname, errs in summary:
        cells = []
        for t in tags:
            m = [r for r in out_rows if r["method"] == mname and r["fusion"] == fname and r["tag_id"] == t]
            if not m:
                cells.append(f"{'-':>18s}")
            elif refs and m[0]["error_m"] != "":
                cells.append(f"{m[0]['north_m']:+.2f},{m[0]['east_m']:+.2f} e{m[0]['error_m']:.2f}".rjust(18))
            else:
                cells.append(f"{m[0]['north_m']:+.2f},{m[0]['east_m']:+.2f} s{m[0]['spread_m']:.2f}".rjust(18))
        tail = ""
        if errs:
            tail = f"  {math.sqrt(np.mean(np.square(errs))):.3f} {max(errs):.3f}"
        print(f"{mname:14s} {fname:13s} " + " ".join(cells) + tail)
    if refs:
        print("\ne = horizontal error to --refs (m).  Lower RMSE = better.  "
              "Pick the method on one run, confirm it on another run before trusting it.")
    else:
        print("\ns = spread of the inlier detections (m).  No --refs: accuracy NOT evaluated.")

    with open(run / "evaluation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]) if out_rows else ["method"])
        w.writeheader()
        w.writerows(out_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 7))
        # flown path: PX4 position from px4_telemetry.csv (10 Hz), in the start frame
        tel = run / "px4_telemetry.csv"
        if tel.exists():
            fn, fe = [], []
            with open(tel, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("px4_armed") not in ("1", "True"):
                        continue
                    x, y = f(r.get("px4_x_m")), f(r.get("px4_y_m"))
                    if x is None or y is None:
                        continue
                    n, e = to_start(x, y, origin)
                    fn.append(n); fe.append(e)
            if fn:
                ax.plot(fe, fn, "-", color="C3", lw=1.6, alpha=.85, label="flown (PX4, armed)")
                ax.plot(fe[-1], fn[-1], "v", color="C3", ms=8, label="landed")
        # planned path: the trajectory file named in summary.json, start frame (x ahead, y right)
        try:
            traj = json.loads((run / "summary.json").read_text())["args"].get("trajectory")
        except (OSError, ValueError, KeyError):
            traj = None
        if traj and traj != "None":
            tp = Path(traj)
            if not tp.exists():
                tp = run.parent.parent / traj
            if tp.exists():
                pl = np.genfromtxt(tp, delimiter=",", names=True)
                ax.plot(pl["y"], pl["x"], "--", color="C7", lw=1.2, label=f"planned ({tp.name})")
        mk = {"firmware-pnp": "o", "checker-pnp": "s", "checker-ray": "^"}
        for mname in methods:
            pts = [r for r in out_rows if r["method"] == mname and r["fusion"] == "inlier-wmean"]
            ax.scatter([p["east_m"] for p in pts], [p["north_m"] for p in pts], marker=mk[mname],
                       facecolors="none", edgecolors={"o": "C0", "s": "C1", "^": "C2"}[mk[mname]], label=mname)
        for t, (n, e) in refs.items():
            ax.plot(e, n, "k+", ms=14, mew=2)
            ax.annotate(f"{t}", (e, n), textcoords="offset points", xytext=(6, 6))
        ax.plot(0, 0, "r*", ms=12, label="start")
        ax.set_xlabel("east / right of start (m)")
        ax.set_ylabel("north / ahead of start (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("tag estimates (+ = tape truth)" if refs else "tag estimates (no truth)")
        fig.savefig(run / "evaluation.png", dpi=130, bbox_inches="tight")
    except Exception as ex:       # plotting is optional
        print("plot skipped:", ex)
    print(f"\nwrote {run / 'evaluation.csv'} and evaluation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
