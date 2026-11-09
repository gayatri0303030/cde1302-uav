#!/usr/bin/env python3
"""Baseline AprilTag logging for the CDE1302 mini project (stages 2-3).

Two modes.  Run from the repo root, inside the conda environment.

  observe   No flight commands at all.  Watches telemetry + camera and logs every tag with
            its estimated position.  Use it on the bench (drone powered, propellers OFF)
            to check the camera -> position maths against a tape measure.

      python3 laptop/run_baseline.py --drone-id 5 --esp-ip 192.168.1.205 --mode observe \
          --refs tags_ref.csv

  fly       CMD_START (arm + take off), upload and play a CSV trajectory, CMD_LAND - the
            same flight sequence as laptop/send_trajectory.py - while logging tags.  Key
            events are printed with wall-clock timestamps.  You must type TRAJ-<id> first.

      python3 laptop/run_baseline.py --drone-id 5 --esp-ip 192.168.1.205 --mode fly \
          --trajectory trajectory/ellipse_ours.csv --refs tags_ref.csv \
          --trial 1 --pass-no 1 --height 0.5 --speed 0.15

Everything is written to logs/<mode>_<timestamp>/ :
  events.jsonl, observations.csv (every detection + why it was rejected), summary.json,
  worksheet_rows.csv (paste into the baseline worksheet), config_used.yaml.

Reference file (--refs): one tag per line, "id,north_m,east_m" (header and # comments ok),
measured from the takeoff point in the local frame: +x = the nose direction, +y = right.

Camera: --show opens a live window with the tags outlined (green = confirmed, yellow = k/3
candidate, red = rejected + reason); --record-video saves the same picture as camera.mp4 in the
run folder.  Neither changes detection or flight.

Emergency: RC pilot first.  Ctrl+C after CMD_START sends CMD_LAND (3x).  With --show, pressing l
in the video window does the same.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # laptop/ on the path, like the repo scripts

import numpy as np                                  # noqa: E402

from comms import CommsNode                         # noqa: E402
from protocol import (                              # noqa: E402
    CMD_LAND, CMD_START, NAV_ARRIVED, NAV_TRAJ, TRAJ_MAX_PTS, CommandPacket,
    build_traj_packets, build_traj_start,
)
from send_trajectory import UPLOAD_GAP_S, load_trajectory   # noqa: E402

from miniproject.camera_worker import CameraWorker  # noqa: E402
from miniproject.pose_history import PoseHistory    # noqa: E402
from miniproject.recorder import Recorder           # noqa: E402
from miniproject.settings import DEFAULT_CONFIG, load_settings  # noqa: E402
from miniproject.tag_map import TagMap              # noqa: E402
from miniproject.tag_pipeline import TagPipeline    # noqa: E402

logging.getLogger().setLevel(logging.WARNING)       # keep the console to our event lines


# ---------------------------------------------------------------------------------------
# Telemetry link
# ---------------------------------------------------------------------------------------
class Link:
    """Latest telemetry packet + a pose history for time-synced lookups."""

    def __init__(self, drone_id: int, history: PoseHistory, pose_latency_s: float, rec=None):
        self.rec = rec
        self.drone_id = drone_id
        self.history = history
        self.pose_latency_s = pose_latency_s
        self.pkt = None
        self.rx_t = 0.0
        self._cond = threading.Condition()

    def callback(self, pkt, _ip) -> None:
        if pkt.drone_id != self.drone_id:
            return
        now = time.monotonic()
        epoch = time.time()
        self.history.add(now - self.pose_latency_s, pkt.ned_x, pkt.ned_y, pkt.heading_rad,
                         z=getattr(pkt, "ned_z", None), vx=getattr(pkt, "ned_vx", None),
                         vy=getattr(pkt, "ned_vy", None), vz=getattr(pkt, "ned_vz", None),
                         epoch=epoch - self.pose_latency_s,
                         q=getattr(pkt, "q_wxyz", None), rates=getattr(pkt, "body_rates", None),
                         pos_boot_ms=getattr(pkt, "px4_pos_boot_ms", None),
                         att_boot_ms=getattr(pkt, "px4_att_boot_ms", None),
                         q_from_px4=getattr(pkt, "q_from_px4", None))
        if self.rec is not None:
            g = lambda v, n=5: "" if v is None else round(float(v), n)
            self.rec.telemetry([f"{epoch:.4f}", f"{epoch - self.pose_latency_s:.4f}", g(pkt.ned_x), g(pkt.ned_y),
                                g(getattr(pkt, "ned_z", None)), g(pkt.heading_rad),
                                g(getattr(pkt, "ned_vx", None)), g(getattr(pkt, "ned_vy", None)),
                                g(getattr(pkt, "ned_vz", None)), getattr(pkt, "px4_pos_age_ms", None) if getattr(pkt, "px4_pos_age_ms", None) is not None else "",
                                getattr(pkt, "px4_att_age_ms", None) if getattr(pkt, "px4_att_age_ms", None) is not None else "",
                                getattr(pkt, "px4_armed", None) if getattr(pkt, "px4_armed", None) is not None else "",
                                getattr(pkt, "nav_state", ""),
                                *[g(v, 6) for v in (getattr(pkt, "q_wxyz", None) or (None,) * 4)],
                                *[g(v, 6) for v in (getattr(pkt, "body_rates", None) or (None,) * 3)],
                                g(getattr(pkt, "roll_rad", None), 6), g(getattr(pkt, "pitch_rad", None), 6),
                                "" if getattr(pkt, "px4_pos_boot_ms", None) is None else pkt.px4_pos_boot_ms,
                                "" if getattr(pkt, "px4_att_boot_ms", None) is None else pkt.px4_att_boot_ms,
                                "" if getattr(pkt, "q_from_px4", None) is None else int(pkt.q_from_px4)])
        with self._cond:
            self.pkt, self.rx_t = pkt, now
            self._cond.notify_all()

    def wait_first(self, timeout_s: float, heard) -> None:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self.pkt is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no telemetry from drone {self.drone_id} on UDP 5005 "
                                       f"(heard drones: {sorted(heard()) or 'none'}) - check "
                                       f"drone ID, laptop IP 192.168.1.(100+ID) and the firewall")
                self._cond.wait(min(0.5, left))

    def age(self) -> float:
        return time.monotonic() - self.rx_t if self.pkt is not None else float("inf")


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------
def load_refs(path: str | None) -> dict[int, tuple[float, float]]:
    refs: dict[int, tuple[float, float]] = {}
    if not path:
        return refs
    for ln, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        try:
            refs[int(parts[0])] = (float(parts[1]), float(parts[2]))
        except (ValueError, IndexError):
            if ln == 1:                              # header row such as "id,x,y"
                continue
            raise SystemExit(f"{path}:{ln}: expected 'id,north_m,east_m', got {raw!r}")
    return refs


def parse_ids(text: str | None) -> list[int]:
    return [int(t) for t in text.replace(",", " ").split()] if text else []


def tag_table(rec: Recorder, tag_map: TagMap, refs: dict, label: str, min_obs: int = 3) -> dict:
    """Print + return the per-tag report (the 'reported positions' the examiner checks)."""
    rows = {}
    n_ok = len(tag_map.confirmed_ids())          # unique confirmed IDs: each counted once
    n_all = len(tag_map.ids())
    rec.event(f"TAG_REPORT_{label}", f"{n_ok} confirmed tag(s)" + (f" (+{n_all - n_ok} unconfirmed)" if n_all > n_ok else ""))
    if not tag_map.ids():
        rec.note("  (no tags accepted)")
    for e in tag_map.estimates():
        row = {"id": e.tag_id, "north_m": round(e.north, 3), "east_m": round(e.east, 3),
               "n_obs": e.n_obs, "n_inliers": e.n_inliers, "std_north_m": round(e.std_north, 3),
               "std_east_m": round(e.std_east, 3), "best_range_m": round(e.best_range_m, 2),
               "mean_down_m": round(e.mean_down, 2),
               "first_detect_epoch": round(e.first_t_epoch, 3),
               "accepted_frame_count": e.accepted_frame_count,
               "confirmed_at_epoch": None if e.confirmed_at is None else round(e.confirmed_at, 3),
               "confirmed": e.confirmed_at is not None}
        text = (f"  tag {e.tag_id:>2}  N {e.north:+.3f}  E {e.east:+.3f}  "
                f"(frames {e.accepted_frame_count}, spread {max(e.std_north, e.std_east):.3f} m, best range {e.best_range_m:.2f} m)")
        if e.confirmed_at is None:
            text += f"   UNCONFIRMED ({e.accepted_frame_count}/{min_obs} frames) - NOT a target"
        if e.tag_id in refs:
            rn, re_ = refs[e.tag_id]
            err = math.hypot(e.north - rn, e.east - re_)
            row.update({"ref_north_m": rn, "ref_east_m": re_, "error_m": round(err, 3),
                        "err_north_m": round(e.north - rn, 3), "err_east_m": round(e.east - re_, 3)})
            text += f"   ref ({rn:+.2f}, {re_:+.2f})  ERROR {err:.3f} m"
        rec.note(text)
        rows[e.tag_id] = row
    return rows


def write_worksheet_rows(rec: Recorder, rows: dict, expected: list[int], args, t_ref: float) -> Path:
    """CSV whose columns match the Baseline sheet: paste straight in."""
    path = rec.out_dir / "worksheet_rows.csv"
    ids = expected or sorted(t for t, r in rows.items() if r.get("confirmed", True))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Trial #", "Pass #", "Height (m)", "Speed (m/s)", "Tag ID", "Detected? (Y/N)",
                    "Est x (m)", "Est y (m)", "Timestamp (s)", "accepted_frame_count", "confirmed_at_time (s)"])
        for tid in ids:
            r = rows.get(tid)
            n = r.get("accepted_frame_count", "") if r else 0
            if r and r.get("confirmed", True):
                ca = r.get("confirmed_at_epoch")
                w.writerow([args.trial, args.pass_no, args.height, args.speed, tid, "Y",
                            r["north_m"], r["east_m"], round(r["first_detect_epoch"] - t_ref, 2),
                            n, "" if ca is None else round(ca - t_ref, 2)])
            else:
                w.writerow([args.trial, args.pass_no, args.height, args.speed, tid, "N", "", "", "", n, ""])
    return path


def send_land(comms: CommsNode, drone_id: int, rec: Recorder, why: str) -> bool:
    ok = comms.send_command(drone_id, CommandPacket(CMD_LAND))
    if not ok:
        rec.event("LAND_COMMAND_FAILED", f"drone {drone_id} IP unknown")
        return False
    rec.event("LAND_COMMAND_SENT", f"drone {drone_id} ({why})")
    for _ in range(2):                                # UDP: repeat, like the repo scripts
        time.sleep(0.1)
        comms.send_command(drone_id, CommandPacket(CMD_LAND))
    return True


def sleep_watch(seconds: float, link: Link, stale_s: float, tick=None) -> None:
    """Sleep, but raise if telemetry goes stale (the caller then lands).  tick() runs every 0.1 s
    (the flight guard) and may raise too."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if link.age() > stale_s:
            raise ConnectionError(f"telemetry stale ({link.age():.1f} s)")
        if tick is not None:
            tick()
        time.sleep(0.1)


def play_trajectory(args, comms, link: Link, rec, pts, dt_ms: int, gtick, flags: int = 0, window=None,
                    yaw_limits=None):
    """Upload pts (N x 3 offsets from where the drone hovers when START arrives) and start it.
    Returns (traj_id, t_start_sent, hover_xy).  Raises if the drone refuses 15 times."""
    import random
    traj_id = random.randint(1, 255)
    packets = build_traj_packets(traj_id, pts)
    if window is not None or yaw_limits is not None:
        start = build_traj_start(traj_id, dt_ms, flags=flags, window=window, yaw_limits=yaw_limits)
    elif flags:
        try:
            start = build_traj_start(traj_id, dt_ms, flags=flags)
        except TypeError:                            # old protocol.py: append the flag byte ourselves
            start = build_traj_start(traj_id, dt_ms) + bytes([flags])
    else:
        start = build_traj_start(traj_id, dt_ms)
    for attempt in range(1, 16):
        for p in packets:
            comms.send_raw(args.drone_id, p)
            time.sleep(UPLOAD_GAP_S)
            gtick()
        # The firmware refuses to start unless PX4's position is < 500 ms old.  PX4 sends it to the
        # ESP32 only ~1 Hz on this drone, so wait for a fresh one (age < 250 ms) before START.
        fresh_by = time.monotonic() + 2.5
        while time.monotonic() < fresh_by:
            age = getattr(link.pkt, "px4_pos_age_ms", None) if link.pkt else None
            if age is None or age < 250:
                break
            time.sleep(0.02)
            gtick()
        t_start_sent = time.monotonic()
        hover_xy = (link.pkt.ned_x, link.pkt.ned_y)
        comms.send_raw(args.drone_id, start)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not (link.pkt and link.pkt.nav_state == NAV_TRAJ):
            time.sleep(0.05)
            gtick()
        if link.pkt and link.pkt.nav_state == NAV_TRAJ:
            return traj_id, t_start_sent, hover_xy
        rec.note(f"attempt {attempt}: trajectory not started (see the ESP32 log)")
    raise RuntimeError("drone refused the trajectory")


class HomeTag:
    """Everything the HOME tag tells us during one flight (see miniproject/homing.py)."""

    def __init__(self, args, S, cam, sightings: list):
        self.args, self.S, self.cam, self.sightings = args, S, cam, sightings
        self.tag_xy = tuple(float(v) for v in args.home_tag_at.split(","))
        self.home_xy = tuple(float(v) for v in args.home_at.split(","))
        self.yaw0 = cam.pipeline.origin[2] if cam.pipeline.origin else 0.0
        self.theta_start = None        # tag edge angle in the body frame at take-off
        self.h_start = None            # true heading (start frame) at take-off
        self.drift_start = None        # (t_epoch, PX4-minus-true north, east) at take-off
        self.drift_end = None          # same, back home before any correction
        self.heading_error_deg = None  # PX4 heading drift measured at the end

    def _theta(self, o):
        from apriltag_host import CAMERA_INTRINSICS as K      # 9 Nov: the calibration
        from miniproject.homing import tag_body_yaw
        return tag_body_yaw(getattr(o, "corners", None), self.S.mount, K.fx, K.fy, K.cx, K.cy)

    def look(self, link, gtick, seconds: float, need: int = 5, settle: float = 0.5, t_from=None):
        """Wait (feeding the guard) until `need` accepted sightings arrive or `seconds` pass."""
        t_from = time.monotonic() + settle if t_from is None else t_from
        deadline = max(time.monotonic(), t_from) + seconds
        while time.monotonic() < deadline:
            sleep_watch(0.2, link, self.S.flight.stale_telemetry_s, tick=gtick)
            if sum(1 for t, _ in list(self.sightings) if t >= t_from) >= need:
                break
        return [o for (t, o) in list(self.sightings) if t >= t_from]

    def fix(self, obs, heading_true: float):
        """Drone position in the start frame from the tag, given its TRUE heading (start frame)."""
        from miniproject.homing import home_fix
        sights = [(o.fwd, o.right, self.yaw0 + heading_true) for o in obs]
        return home_fix(sights, self.tag_xy, self.home_xy, self.yaw0) if len(sights) >= 3 else None

    def start(self, link, gtick, rec, cam, t_from=None, when="after take-off"):
        """After take-off: where the drone really is, and the tag's angle, before the path starts."""
        from miniproject.homing import median_angle, wrap
        t_from = time.monotonic() - 3.0 if t_from is None else t_from
        obs = self.look(link, gtick, self.args.home_look_s, t_from=t_from)
        if len(obs) < 3:
            rec.event("HOME_TAG_NOT_SEEN", f"{when}: home tag {self.args.home_tag} not seen "
                      f"({len(obs)} sightings) - using PX4 only")
            return None
        px4_yaw = median_angle([o.pose_yaw for o in obs])
        self.h_start = wrap(px4_yaw - self.yaw0)             # PX4 heading is fresh here: trust it
        self.theta_start = median_angle([self._theta(o) for o in obs])
        f = self.fix(obs, self.h_start)
        px = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
        self.drift_start = (time.time(), px[0] - f.drone_x, px[1] - f.drone_y)
        rec.event("HOME_ANCHOR", f"home tag {self.args.home_tag} seen {when} ({f.n} sightings): drone is at "
                  f"({f.drone_x:+.2f}, {f.drone_y:+.2f}) from the origin, PX4 says ({px[0]:+.2f}, {px[1]:+.2f})",
                  drone_x=f.drone_x, drone_y=f.drone_y, px4_x=px[0], px4_y=px[1],
                  tag_angle_deg=None if self.theta_start is None else math.degrees(self.theta_start))
        return f

    def heading_now(self, obs, link):
        """True heading (start frame): take-off heading + change seen in the tag's angle."""
        from miniproject.homing import median_angle, wrap
        px4_h = wrap(link.pkt.heading_rad - self.yaw0)
        th = median_angle([self._theta(o) for o in obs])
        if self.h_start is None or self.theta_start is None or th is None:
            return px4_h, None
        h_true = wrap(self.h_start + wrap(self.theta_start - th))
        return h_true, math.degrees(wrap(px4_h - h_true))


def rezero_from_home(cam, rec, tag_map, state: dict, home: HomeTag, home_id: int) -> None:
    """Re-set (0, 0) from the home tag once the drone hovers after take-off.  PX4 keeps its own
    numbers, so the laptop shifts its zero instead: from now on every position (live tag
    positions, worksheet, 'pos from start') is measured from the TRUE start spot the tag gives
    (tag position minus its 0.45 m), not from where PX4 thinks the start is."""
    t, dn, de = home.drift_start                 # PX4 minus true, start frame (m)
    x0, y0, yaw0 = cam.pipeline.origin
    c, s = math.cos(yaw0), math.sin(yaw0)
    n_targets = len(state["targets"])            # sightings so far: move them onto the new zero
    for i in range(n_targets):
        tt, tid, n, e = state["targets"][i]
        state["targets"][i] = (tt, tid, n - dn, e - de)
    tag_map.shift(dn, de)
    cam.pipeline.set_origin(x0 + dn * c - de * s, y0 + dn * s + de * c, yaw0)
    home.drift_start = (t, 0.0, 0.0)
    rec.event("ORIGIN_REZEROED", f"(0, 0) re-set from home tag {home_id}: PX4 was ({dn:+.2f}, {de:+.2f}) m off "
              f"after take-off; all positions now measured from the tag-based zero",
              px4_offset_north=dn, px4_offset_east=de)


def settled_position(link: Link, cam, f, gtick, settle_s: float = 0.0, samples: int = 1):
    """Where the drone is (start frame: ahead of the take-off nose, right) after settle_s of hovering.
    PX4 sends its position only ~1 Hz on this drone and the hover wanders a few cm, so this waits
    for `samples` NEW fresh PX4 samples and returns (mean north, mean east, latest north, latest east)."""
    if settle_s > 0:
        sleep_watch(settle_s, link, f.stale_telemetry_s, tick=gtick)
    got = []
    b_last = getattr(link.pkt, "px4_pos_boot_ms", None)
    deadline = time.monotonic() + 1.6 * samples + 1.0
    while len(got) < samples and time.monotonic() < deadline:
        p = link.pkt
        age = getattr(p, "px4_pos_age_ms", None)
        b = getattr(p, "px4_pos_boot_ms", None)
        if (b is None or b != b_last) and (age is None or age < 300):
            got.append(cam.pipeline._to_origin(p.ned_x, p.ned_y))
            b_last = b
            if b is None:                              # old firmware: no sample counter, space them out
                sleep_watch(1.0, link, f.stale_telemetry_s, tick=gtick)
        sleep_watch(0.05, link, f.stale_telemetry_s, tick=gtick)
    if not got:
        got = [cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)]
    mn = sum(g[0] for g in got) / len(got)
    me = sum(g[1] for g in got) / len(got)
    return mn, me, got[-1][0], got[-1][1]


def _clamp(v, limit: float):
    n = math.hypot(v[0], v[1])
    return (v[0], v[1]) if n <= limit else (v[0] * limit / n, v[1] * limit / n)


def go_to_start(args, S, comms, link: Link, cam, rec, guard, gtick, tag_map, state: dict,
                home, start_fix):
    """After take-off, fly back onto (0, 0) and only return once the drone is there (within
    --start-tolerance, judged on the mean of a few PX4 samples), so the trajectory starts from the
    start mark.  Works with or without tag 20: the position is PX4's, measured from the start; when
    tag 20 was seen it has already re-set that zero (ORIGIN_REZEROED), so it is the tag's position.
    If tag 20 was NOT seen at take-off, the drone looks for it once more after the first move.

    Drone 5 settles ~0.1-0.2 m off every target, the same way each time (2 Nov E8a: always ahead
    and left).  Sending it to (0, 0) again cannot fix that, so each move aims PAST (0, 0) by what the
    last one missed (aim -= gain * miss).  The aim point is kept in state["start_aim"]; the path and
    the home corrections are flown relative to it.  Returns True if at least one move was made."""
    from miniproject.homing import move_path, start_to_ned
    f = S.flight
    dt_ms = round(f.dt_s * 1000)
    tol, tries, gain = args.start_tolerance, args.start_tries, args.aim_gain
    yaw0 = cam.pipeline.origin[2]
    src = f"tag {args.home_tag}" if start_fix is not None else "PX4 only (tag not seen yet)"
    rec.event("START_RETURN", f"going back to (0, 0) before the trajectory: within {tol:.2f} m, "
              f"at most {tries} moves; position from {src}")
    moves, relooked = 0, start_fix is not None or home is None
    aim = (0.0, 0.0)                                  # where to SEND the drone so it ends on (0, 0)
    n0, e0, ln, le = settled_position(link, cam, f, gtick, samples=2)
    while True:
        d = math.hypot(n0, e0)
        rec.note(f"start: drone at ({n0:+.2f}, {e0:+.2f}) m from (0, 0), {d:.2f} m off"
                 + (f"; aiming at ({aim[0]:+.2f}, {aim[1]:+.2f})" if moves else ""))
        if not relooked and moves >= 1:               # back near the start: tag 20 should be in view now
            relooked = True
            fix = home.start(link, gtick, rec, cam, t_from=time.monotonic(), when="back over the start")
            if fix is not None:
                _, dn, de = home.drift_start                 # positions shift by -(dn, de): so does the aim
                rezero_from_home(cam, rec, tag_map, state, home, args.home_tag)
                state["home_drift_start"] = home.drift_start
                aim = (aim[0] - dn, aim[1] - de)
                n0, e0, ln, le = settled_position(link, cam, f, gtick, samples=2)
                continue
        done = d <= tol
        if done or moves >= tries:
            state["start_aim"] = aim
            state["start_return"] = {"north_m": round(n0, 3), "east_m": round(e0, 3),
                                     "distance_m": round(d, 3), "moves": moves, "reached": done,
                                     "aim_north_m": round(aim[0], 3), "aim_east_m": round(aim[1], 3)}
            if done:
                rec.event("START_REACHED", f"at ({n0:+.2f}, {e0:+.2f}) m, {d:.2f} m from (0, 0) after {moves} "
                          f"move(s) - starting the trajectory", north=n0, east=e0, moves=moves)
            else:
                rec.event("START_NOT_REACHED", f"still {d:.2f} m from (0, 0) after {moves} moves - starting "
                          f"anyway, with the path shifted so it begins on (0, 0)", north=n0, east=e0, moves=moves)
            if moves:
                rec.note(f"settle offset learnt: the drone stops about ({-aim[0]:+.2f}, {-aim[1]:+.2f}) m from "
                         f"where it is sent; the path and the home corrections allow for it")
            return moves > 0
        if d > args.start_max_move:
            raise RuntimeError(f"drone is {d:.2f} m from (0, 0) after take-off (more than --start-max-move "
                               f"{args.start_max_move:g} m) - not flying the trajectory")
        if moves >= 1:                                # it settled `(n0, e0)` off the last aim: aim past it
            aim = _clamp((aim[0] - gain * n0, aim[1] - gain * e0), args.aim_max)
        moves += 1
        mx, my = aim[0] - ln, aim[1] - le             # from where PX4 says it is now to the aim point
        n, e = start_to_ned(mx, my, yaw0)
        pts = move_path(n, e, f.dt_s)
        rec.event("START_MOVE", f"move {moves}: ({mx:+.2f}, {my:+.2f}) m (ahead, right) in "
                  f"{len(pts) * f.dt_s:.1f} s, aiming at ({aim[0]:+.2f}, {aim[1]:+.2f})")
        _, t0, hover_xy = play_trajectory(args, comms, link, rec, pts, dt_ms, gtick)

        def expected_xy(t_mono: float, t0=t0, hover_xy=hover_xy, pts=pts):
            k = min(max(int(round((t_mono - t0) / f.dt_s)), 0), len(pts) - 1)
            return hover_xy[0] + float(pts[k, 0]), hover_xy[1] + float(pts[k, 1])
        guard.start_trajectory(expected_xy)            # stays on (expects the end point) while it settles
        end = time.monotonic() + len(pts) * f.dt_s + 8.0
        while link.pkt.nav_state == NAV_TRAJ and time.monotonic() < end:
            sleep_watch(0.1, link, f.stale_telemetry_s, tick=gtick)
        if link.pkt.nav_state != NAV_ARRIVED:
            raise RuntimeError(f"move back to (0, 0) ended in nav state {link.pkt.nav_state_name}")
        n0, e0, ln, le = settled_position(link, cam, f, gtick, settle_s=args.start_settle_s,
                                          samples=args.start_samples)


def return_home(args, S, comms, link: Link, cam, rec, guard, gtick, home: HomeTag, state: dict) -> None:
    """After the trajectory: use the HOME tag to land on the origin (see homing.py).
    Not seen -> short search: 0.3 m forward (tag was too far ahead), then 0.3 m behind the start
    point (tag was too close / under the camera), then give up and land where it is."""
    from miniproject.homing import move_path, start_to_ned, wrap
    f = S.flight
    dt_ms = round(f.dt_s * 1000)
    rec.event("HOME_RETURN", f"looking for home tag {args.home_tag} (tape: {home.tag_xy[0]:+.2f}, "
              f"{home.tag_xy[1]:+.2f}); landing spot ({home.home_xy[0]:+.2f}, {home.home_xy[1]:+.2f})")

    def move(dx: float, dy: float, why: str, h_true: float) -> bool:
        """Move by (dx, dy) in the start frame.  PX4's axes are turned by its heading error, so the
        offset is rotated with PX4's heading minus the TRUE heading, not the take-off heading."""
        n, e = start_to_ned(dx, dy, link.pkt.heading_rad - h_true)
        pts = move_path(n, e, f.dt_s)
        rec.event("HOME_MOVE", f"{why}: ({dx:+.2f}, {dy:+.2f}) m (ahead, right) in {len(pts) * f.dt_s:.1f} s")
        _, t0, hover_xy = play_trajectory(args, comms, link, rec, pts, dt_ms, gtick)

        def expected_xy(t_mono: float):
            k = min(max(int(round((t_mono - t0) / f.dt_s)), 0), len(pts) - 1)
            return hover_xy[0] + float(pts[k, 0]), hover_xy[1] + float(pts[k, 1])
        guard.start_trajectory(expected_xy)
        end = time.monotonic() + len(pts) * f.dt_s + 8.0
        while link.pkt.nav_state == NAV_TRAJ and time.monotonic() < end:
            sleep_watch(0.1, link, f.stale_telemetry_s, tick=gtick)
        guard.stop()
        if link.pkt.nav_state != NAV_ARRIVED:
            rec.note(f"home: move ended in nav state {link.pkt.nav_state_name} - landing")
            return False
        return True

    searches = [(0.30, 0.0, "search: tag may be too far ahead"),
                (-0.60, 0.0, "search: tag may be too close")]
    corrections = 0
    aim = tuple(state.get("start_aim") or (0.0, 0.0))   # settle offset learnt before the path
    first = True
    for _ in range(args.home_tries + len(searches) + 1):
        obs = home.look(link, gtick, args.home_look_s)
        if len(obs) < 3:
            if not searches:
                rec.event("HOME_TAG_NOT_SEEN", f"home tag {args.home_tag} not found ({len(obs)} sightings) - "
                          f"landing where the drone is")
                return
            dx, dy, why = searches.pop(0)
            if not move(dx + aim[0], dy + aim[1], why, wrap(link.pkt.heading_rad - home.yaw0)):
                return
            continue
        searches = []                                        # found it: no more searching
        h_true, h_err = home.heading_now(obs, link)
        fix = home.fix(obs, h_true)
        px = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
        if first:
            first = False
            home.heading_error_deg = h_err
            state["home_drift_end"] = home.drift_end = (time.time(), px[0] - fix.drone_x, px[1] - fix.drone_y)
        herr = "" if h_err is None else f"; PX4 heading is {h_err:+.1f} deg off (nose {-h_err:+.1f} deg vs take-off)"
        rec.event("HOME_TAG_SEEN", f"drone is at ({fix.drone_x:+.2f}, {fix.drone_y:+.2f}) by the home tag "
                  f"({fix.n} sightings, spread {fix.spread_m:.2f} m); PX4 thinks ({px[0]:+.2f}, {px[1]:+.2f}); "
                  f"{fix.distance:.2f} m from the landing spot{herr}",
                  drone_x=fix.drone_x, drone_y=fix.drone_y, px4_x=px[0], px4_y=px[1], error_m=fix.distance,
                  heading_error_deg=h_err)
        if fix.distance <= args.home_tolerance:
            rec.event("HOME_REACHED", f"within {fix.distance:.2f} m of the landing spot - landing")
            return
        if corrections >= args.home_tries:
            rec.note(f"home: still {fix.distance:.2f} m off after {corrections} corrections - landing here")
            return
        if fix.distance > args.home_max_move:
            rec.note(f"home: correction {fix.distance:.2f} m is more than --home-max-move {args.home_max_move} m"
                     f" - not trusting it, landing here")
            return
        if corrections >= 1:                     # the last correction stopped short: aim past the spot
            aim = _clamp((aim[0] + args.aim_gain * fix.move_x, aim[1] + args.aim_gain * fix.move_y), args.aim_max)
        corrections += 1
        mx, my = fix.move_x + aim[0], fix.move_y + aim[1]
        if not move(mx, my, f"correction {corrections}" + (f" (aiming {aim[0]:+.2f}, {aim[1]:+.2f} past the spot)"
                                                          if any(aim) else ""), h_true):
            return


def drift_corrected_table(rec, refs: dict, target_obs: list, d0, d1, home_id) -> dict:
    """Tag positions with PX4's drift removed: the home tag measured the drift at take-off (d0)
    and when the drone came back (d1); in between it is taken to grow linearly with time."""
    import numpy as np
    t0, n0, e0 = d0
    t1, n1, e1 = d1
    span = max(1e-3, t1 - t0)
    per = {}
    for t, tid, n, e in target_obs:
        k = min(max((t - t0) / span, 0.0), 1.0)
        per.setdefault(tid, []).append((n - (n0 + k * (n1 - n0)), e - (e0 + k * (e1 - e0))))
    rec.event("HOME_DRIFT_CORRECTED", f"PX4 drift by the home tag: ({n0:+.2f}, {e0:+.2f}) m at take-off -> "
              f"({n1:+.2f}, {e1:+.2f}) m back home ({span:.0f} s); tag positions with that drift removed:")
    rows = {}
    for tid in sorted(per):
        v = np.array(per[tid])
        cn, ce = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
        row = {"id": tid, "north_m": round(cn, 3), "east_m": round(ce, 3), "n_obs": len(v)}
        text = f"  tag {tid:>2}  N {cn:+.3f}  E {ce:+.3f}  ({len(v)} sightings)"
        if tid in refs:
            rn, re_ = refs[tid]
            err = math.hypot(cn - rn, ce - re_)
            row.update(ref_north_m=rn, ref_east_m=re_, error_m=round(err, 3))
            text += f"   ref ({rn:+.2f}, {re_:+.2f})  ERROR {err:.3f} m"
        rec.note(text)
        rows[tid] = row
    return rows


def home_drift_start_or_zero(state: dict, rec, home_id):
    """Drift at take-off for the tag-20 correction.  Normally measured after take-off (and then 0,
    because the origin was re-zeroed on it).  If the home tag was missed after take-off but seen
    back home, take PX4 as right at take-off, so the drift measured at the end is still removed."""
    if state.get("home_drift_start"):
        return state["home_drift_start"]
    if home_id is None or not state.get("home_drift_end") or not state.get("t_takeoff"):
        return None
    rec.note(f"home tag {home_id} was not seen after take-off: drift at take-off taken as 0, "
             f"only the drift measured back home is removed")
    return (state["t_takeoff"], 0.0, 0.0)


def _deviation(xy, t_rel, plan, dt, window_s=5.0):
    """Distance from each flown point to the GIVEN path, searching the plan only within +-window_s
    of where the plan says the drone should be at that moment (so a lane next door is never
    mistaken for the right one).  Returns (cross-track distances, distances to the timed point)."""
    import numpy as np
    n = len(plan)
    w = max(1, int(round(window_s / dt)))
    cross, timed = [], []
    for (x, y), t in zip(xy, t_rel):
        k = int(round(t / dt))
        k = min(max(k, 0), n - 1)
        seg = plan[max(0, k - w): min(n, k + w + 1)]
        cross.append(float(np.min(np.hypot(seg[:, 0] - x, seg[:, 1] - y))))
        timed.append(float(np.hypot(plan[k, 0] - x, plan[k, 1] - y)))
    return np.array(cross), np.array(timed)


def flight_metrics(rec, state: dict, rows: dict, refs: dict, d0, d1, out_dir: Path,
                   yaw_follow: bool = False) -> dict:
    """The trajectory experiment's three measurements, for every fly run:
      1. mission time      take-off command -> land command (and the trajectory part alone)
      2. tags detected     confirmed target tags (of the ones in --refs)
      3. deviation         flown path vs the GIVEN path (the CSV), in the start frame:
                           by PX4's own position, and - when the home tag measured the drift -
                           with that drift removed (closer to where the drone really was)
    Written to trajectory_metrics.json, path_vs_plan.png and a TRAJECTORY_METRICS line."""
    import json as _json
    import numpy as np
    m: dict = {"completed": bool(state.get("t_land")), "yaw_follow_requested": yaw_follow}
    if state.get("headings"):                       # how far the nose actually turned on the path
        hh = np.unwrap(np.array(state["headings"]))
        m["heading_swept_deg"] = round(float(np.degrees(hh.max() - hh.min())), 1)
    if state.get("t_takeoff") and state.get("t_land"):
        m["mission_time_s"] = round(state["t_land"] - state["t_takeoff"], 1)
    if state.get("t_traj_start") and state.get("t_traj_end"):
        m["trajectory_time_s"] = round(state["t_traj_end"] - state["t_traj_start"], 1)
    plan, dt = state.get("plan_xy"), state.get("plan_dt", 0.05)
    given = plan
    sent = state.get("plan_sent_xy")                 # 7 Nov 19:10: the path as sent (anchor-fix shifts)
    if plan is not None and sent is not None and len(sent) == len(plan):
        plan = sent                                  # PX4 is told to follow this one: deviation against it
    if plan is not None:
        m["planned_time_s"] = round(len(plan) * dt, 1)
        m["planned_length_m"] = round(float(np.sum(np.hypot(*np.diff(plan, axis=0).T))), 2)
    rows = rows or {}
    conf = sorted(int(k) for k, r in rows.items() if r.get("confirmed") and int(k) in refs)
    m["tags_total"] = len(refs)
    m["tags_detected"] = len(conf)
    m["tags_detected_ids"] = conf
    m["tags_seen_not_confirmed"] = sorted(int(k) for k, r in rows.items() if not r.get("confirmed"))
    track = state.get("track") or []
    t0 = rec.t0
    px4 = corr = t_rel = None
    if plan is not None and track and t0 and state.get("t_traj_start"):
        pts, last = [], None
        for t, n, e, rx, ry in track:                # one point per NEW PX4 position (real: 1 Hz)
            if last is None or (rx, ry) != last:
                pts.append((t0 + t, n, e))
                last = (rx, ry)
        a = np.array(pts)
        t_end = state.get("t_traj_end") or a[-1, 0]
        a = a[(a[:, 0] >= state["t_traj_start"] - 0.5) & (a[:, 0] <= t_end + 0.5)]   # the planned piece only
        t_rel = a[:, 0] - state["t_traj_start"]
        px4 = a[:, 1:3]
        cross, timed = _deviation(px4, t_rel, plan, dt)
        m["deviation_px4_mean_m"] = round(float(cross.mean()), 3)
        m["deviation_px4_max_m"] = round(float(cross.max()), 3)
        m["samples"] = int(len(px4))
        src, use, use_timed = "PX4 only (home tag not seen back home)", cross, timed
        if d0 and d1:
            (ta, na, ea), (tb, nb, eb) = d0, d1
            k = np.clip((a[:, 0] - ta) / max(1e-3, tb - ta), 0.0, 1.0)
            corr = px4 - np.c_[na + k * (nb - na), ea + k * (eb - ea)]
            use, use_timed = _deviation(corr, t_rel, plan, dt)
            src = "PX4 with the home-tag drift removed"
            m["deviation_corrected_mean_m"] = round(float(use.mean()), 3)
            m["deviation_corrected_max_m"] = round(float(use.max()), 3)
        m["deviation_mean_m"] = round(float(use.mean()), 3)
        m["deviation_max_m"] = round(float(use.max()), 3)
        m["deviation_timed_mean_m"] = round(float(use_timed.mean()), 3)
        m["deviation_source"] = src
        fl = corr if corr is not None else px4
        # 6 Nov: everything AFTER the main loop too - backup loop, way back, landing - and where it touched down
        b = np.array(pts)
        after = b[b[:, 0] > t_end + 0.5]
        if len(after):
            rest = after[:, 1:3]
            if d0 and d1:
                (ta, na, ea), (tb, nb, eb) = d0, d1
                k2 = np.clip((after[:, 0] - ta) / max(1e-3, tb - ta), 0.0, 1.0)
                rest = rest - np.c_[na + k2 * (nb - na), ea + k2 * (eb - ea)]
            state["_after_loop"] = np.vstack([fl[-1:], rest]) if len(fl) else rest
            m["touchdown_px4_m"] = [round(float(v), 3) for v in after[-1, 1:3]]
            m["touchdown_est_m"] = [round(float(v), 3) for v in rest[-1]]
        m["flown_length_m"] = round(float(np.sum(np.hypot(*np.diff(fl, axis=0).T))), 2)
    (out_dir / "trajectory_metrics.json").write_text(_json.dumps(m, indent=1))
    mt = f"{m['mission_time_s']:.1f} s" if "mission_time_s" in m else "n/a (not landed normally)"
    tt = f" (trajectory {m['trajectory_time_s']:.1f} s of a planned {m.get('planned_time_s', 0):.1f} s)" \
        if "trajectory_time_s" in m else ""
    dv = (f"mean {m['deviation_mean_m']:.3f} m, max {m['deviation_max_m']:.3f} m ({m['deviation_source']}"
          + (f"; PX4's own view mean {m['deviation_px4_mean_m']:.3f} m" if corr is not None else "") + ")"
          if "deviation_mean_m" in m else "n/a")
    yw = ""
    if "heading_swept_deg" in m:
        yw = f" | nose turned {m['heading_swept_deg']:.0f} deg" + (" (yaw-follow)" if yaw_follow else "")
    rec.event("TRAJECTORY_METRICS", f"mission time {mt}{tt} | tags detected {m['tags_detected']}/{m['tags_total']} "
              f"{m['tags_detected_ids']} | deviation from the given path: {dv}{yw}", **{
                  k: v for k, v in m.items() if not isinstance(v, (list, dict))})
    if px4 is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches
            fig, ax = plt.subplots(figsize=(5.6, 7.0))
            ax.plot(given[:, 1], given[:, 0], "--", color="k", lw=1.2, zorder=5, label="given path")
            if plan is not given:
                ax.plot(plan[:, 1], plan[:, 0], ":", color="#59a14f", lw=1.4, zorder=5,
                        label="path sent to PX4 (anchor fix)")
            ax.plot(px4[:, 1], px4[:, 0], ".-", color="#f28e2b", ms=3, lw=1, label="PX4's position")
            if corr is not None:
                ax.plot(corr[:, 1], corr[:, 0], ".-", color="#4e79a7", ms=3, lw=1, label="home-tag (23) drift removed")
            aft = state.get("_after_loop")
            if aft is not None and len(aft):
                ax.plot(aft[:, 1], aft[:, 0], ".-", color="#b07aa1", ms=3, lw=1,
                        label="after the loop: backup, way back, landing")
                ax.plot(aft[-1, 1], aft[-1, 0], "X", color="#b07aa1", ms=10, zorder=6)
                ax.annotate(f"touchdown ({aft[-1, 0]:+.2f}, {aft[-1, 1]:+.2f})", (aft[-1, 1], aft[-1, 0]),
                            textcoords="offset points", xytext=(8, -12), fontsize=8, color="#b07aa1")
            ax.add_patch(matplotlib.patches.Rectangle((-0.45, -0.45), 0.9, 0.9, fill=False, ec="#59a14f", lw=1.2))
            for tid, (rn, re_) in refs.items():
                ok = tid in conf
                ax.plot(re_, rn, "s", ms=9, mfc="#59a14f" if ok else "white", mec="#59a14f" if ok else "#e15759")
                ax.annotate(str(tid), (re_, rn), textcoords="offset points", xytext=(7, -3), fontsize=8)
            ax.plot(0, 0, "ko", ms=5)
            ax.set_aspect("equal")
            ax.grid(alpha=.3)
            ax.set_xlabel("right of the start (m)")
            ax.set_ylabel("ahead of the start (m)")
            ax.set_title(f"mission {mt}, tags {m['tags_detected']}/{m['tags_total']}\n"
                         f"deviation mean {m.get('deviation_mean_m', float('nan')):.3f} m, "
                         f"max {m.get('deviation_max_m', float('nan')):.3f} m", fontsize=10)
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / "path_vs_plan.png", dpi=120)
            plt.close(fig)
        except Exception as exc:                  # noqa: BLE001 - plot is optional
            rec.note(f"path_vs_plan.png skipped: {exc}")
    return m


def write_home_corrected_csv(path: Path, rows: dict) -> None:
    """Tag table after the tag-20 drift correction (same frame and columns as results.csv style)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("tag_id,north_m,east_m,n_obs,ref_north_m,ref_east_m,error_m\n")
        for tid in sorted(rows):
            r = rows[tid]
            fh.write(f"{tid},{r['north_m']},{r['east_m']},{r['n_obs']},{r.get('ref_north_m', '')},"
                     f"{r.get('ref_east_m', '')},{r.get('error_m', '')}\n")


def make_guard(S, cam, link: Link, rec):
    """Flight guard fed from the latest telemetry (see miniproject/flight_guard.py)."""
    from miniproject.flight_guard import FlightGuard
    f = S.flight
    o = cam.pipeline.origin or (link.pkt.ned_x, link.pkt.ned_y, 0.0)
    guard = FlightGuard((o[0], o[1]), max_jump_m=f.max_jump_m, max_jump_speed_mps=f.max_jump_speed_mps,
                        takeoff_max_drift_m=f.takeoff_max_drift_m, max_path_error_m=f.max_path_error_m,
                        path_error_samples=f.path_error_samples, enabled=f.guard_enabled)

    def tick() -> None:
        p = link.pkt
        if p is None:
            return
        age = getattr(p, "px4_pos_age_ms", None)
        t_sample = link.rx_t - (age / 1000.0 if age is not None else 0.0)
        guard.update(p.ned_x, p.ned_y, t_sample, getattr(p, "px4_pos_boot_ms", None))

    if f.guard_enabled:
        rec.note(f"flight guard on: lands if PX4's position jumps > {f.max_jump_m:g} m (> {f.max_jump_speed_mps:g} m/s), "
                 f"drifts > {f.takeoff_max_drift_m:g} m during take-off, or is > {f.max_path_error_m:g} m off the path")
    else:
        rec.note("flight guard OFF (flight.guard_enabled = false)")
    return guard, tick


# ---------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------
OFFSET_CLEARED_AT = 0.0      # monotonic time when the start offset was reset to (0, 0)


def set_origin(cam: CameraWorker, link: Link, rec: Recorder) -> None:
    """Tag positions are reported from where the drone is NOW, with +North along its nose."""
    # Wait for telemetry sent AFTER the start offset was cleared, or the origin would include the old offset.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not (link.rx_t > OFFSET_CLEARED_AT + 0.4):
        time.sleep(0.05)
    p = link.pkt
    cam.pipeline.set_origin(p.ned_x, p.ned_y, p.heading_rad)
    rec.event("ORIGIN_SET", f"tag positions are measured from here: PX4 local ({p.ned_x:+.2f}, {p.ned_y:+.2f}), "
              f"heading {math.degrees(p.heading_rad):+.0f} deg (+North = nose)",
              origin_x=p.ned_x, origin_y=p.ned_y, origin_yaw_rad=p.heading_rad)
    if math.hypot(p.ned_x, p.ned_y) > 0.30:
        rec.note("  note: PX4 does not read (0, 0) here (drone not power-cycled since the last flight). "
                 "That is fine: refs.csv must be measured from THIS spot, nose = +North.")


def run_observe(args, S, link: Link, cam: CameraWorker, rec: Recorder, tag_map: TagMap) -> None:
    set_origin(cam, link, rec)
    guard, gtick = make_guard(S, cam, link, rec)
    tag_map.reset()                                  # new mission: no counts, no confirmations
    rec.set_t0()
    rec.event("OBSERVE_START", "no flight commands will be sent; Ctrl+C to stop")
    end = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    next_status = 0.0
    while time.monotonic() < end:
        now = time.monotonic()
        if now >= next_status:
            next_status = now + 5.0
            p = link.pkt
            if p:
                rn, re_ = cam.pipeline._to_origin(p.ned_x, p.ned_y)
                o = cam.pipeline.origin
                rh = math.degrees((p.heading_rad - o[2] + math.pi) % (2 * math.pi) - math.pi) if o else math.degrees(p.heading_rad)
                pos = f"pose from start ({rn:+.2f}, {re_:+.2f}) hdg {rh:+.0f} deg [PX4 raw ({p.ned_x:+.2f}, {p.ned_y:+.2f})]"
            else:
                pos = "no telemetry"
            mv = getattr(rec, "mav", None)
            mtxt = f", {mv.summary()}" if mv is not None else ""
            rec.note(f"status: camera {cam.fps:.1f} fps, frame age {cam.last_age_ms} ms, {pos}, "
                     f"confirmed {tag_map.confirmed_ids()} candidates "
                     f"{[t for t in tag_map.ids() if not tag_map.is_confirmed(t)] or '-'}{mtxt}")
        time.sleep(0.2)


def run_fly(args, S, comms: CommsNode, link: Link, cam: CameraWorker, rec: Recorder,
            tag_map: TagMap, refs: dict, state: dict) -> None:
    f = S.flight
    pts = load_trajectory(args.trajectory, f.dt_s)
    side = {}
    side_path = Path(args.trajectory).with_suffix(".json")
    if side_path.exists():                          # make_trajectory.py notes (outward_rectangle)
        import json as _json
        try:
            side = _json.loads(side_path.read_text())
        except ValueError:
            side = {}
    state["traj_info"] = side
    window = None
    if side.get("nose_outward_from") is not None:
        window = (int(side["nose_outward_from"]), int(side["nose_outward_to"]))
        rec.note(f"outward path: nose along the path out and back, turned 90 deg right (outward) for points "
                 f"{window[0]}-{window[1]} ({window[0] * f.dt_s:.1f}-{window[1] * f.dt_s:.1f} s); lands as soon "
                 f"as it is back (needs the yaw v2 firmware)")
    state["plan_xy"] = pts[:, :2].copy()           # the GIVEN path (start frame), for the deviation
    state["plan_dt"] = f.dt_s
    if args.height > 0.5:
        # NED: negative z = up.  Smooth 6 s climb at the start and 6 s descent at the end.
        dz = -(args.height - 0.5)
        t = np.arange(len(pts)) * f.dt_s
        T = t[-1]
        ramp = min(6.0, T / 3)
        k = np.ones_like(t)
        k = np.where(t < ramp, 0.5 - 0.5 * np.cos(np.pi * t / ramp), k)
        k = np.where(t > T - ramp, 0.5 - 0.5 * np.cos(np.pi * (T - t) / ramp), k)
        pts = pts.copy()
        pts[:, 2] = pts[:, 2] + dz * k
        rec.note(f"height {args.height:.2f} m: climb {-dz:.2f} m over the first {ramp:.0f} s, back down over "
                 f"the last {ramp:.0f} s; tag range limit {S.gate.max_range_m:.2f} m")
    dt_ms = round(f.dt_s * 1000)
    duration = (len(pts) - 1) * f.dt_s
    vmax = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).max() / f.dt_s)
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    print(f"trajectory {args.trajectory}: {len(pts)} points, {duration:.1f} s, max speed {vmax:.3f} m/s, "
          f"extent N[{pts[:, 0].min():+.2f},{pts[:, 0].max():+.2f}] E[{pts[:, 1].min():+.2f},{pts[:, 1].max():+.2f}] "
          f"z[{zmin:+.2f},{zmax:+.2f}]")
    if vmax > f.max_speed_mps + 1e-6:
        raise SystemExit(f"max speed {vmax:.3f} m/s exceeds the cap {f.max_speed_mps} m/s (config flight.max_speed_mps)")
    if len(pts) > TRAJ_MAX_PTS:
        raise SystemExit(f"{len(pts)} points > {TRAJ_MAX_PTS}; shorten the trajectory")
    if zmax > 0.05 or zmin < -0.6:
        raise SystemExit(f"z range [{zmin:+.2f},{zmax:+.2f}] m is outside the allowed [-0.60, +0.05] "
                         f"(NED: negative = up from the 0.5 m hover)")

    answer = input(f"Type TRAJ-{args.drone_id} to take off and fly {args.trajectory}: ").strip()
    if answer != f"TRAJ-{args.drone_id}":
        rec.note("confirmation did not match - nothing sent")
        return

    # ---- takeoff ----------------------------------------------------------------------
    set_origin(cam, link, rec)
    guard, gtick = make_guard(S, cam, link, rec)
    state["guard"] = guard
    tag_map.reset()                                  # new mission: no counts, no confirmations
    rec.set_t0()
    if not comms.send_command(args.drone_id, CommandPacket(CMD_START)):
        raise RuntimeError(f"CMD_START not sent: drone {args.drone_id} IP unknown")
    state["start_sent"] = True
    state["t_takeoff"] = time.time()
    rec.event("TAKEOFF_COMMAND_SENT", f"drone {args.drone_id}: CMD_START (arm + take off to 0.5 m)")
    sleep_watch(f.takeoff_wait_s, link, f.stale_telemetry_s, tick=gtick)
    home = start_fix = None
    if args.home_tag is not None:
        home = HomeTag(args, S, cam, state["home_sightings"])
        state["home"] = home
        start_fix = home.start(link, gtick, rec, cam)
        if start_fix is not None:
            rezero_from_home(cam, rec, tag_map, state, home, args.home_tag)
            state["home_drift_start"] = home.drift_start
    # ---- (--start-return) back onto (0, 0) before the path, tag 20 seen or not ---------------
    if getattr(args, "start_return", False) and cam.pipeline.origin is not None and link.pkt is not None:
        state["start_moved"] = go_to_start(args, S, comms, link, cam, rec, guard, gtick, tag_map,
                                           state, home, start_fix)

    # ---- upload + play ----------------------------------------------------------------------
    # The firmware adds the CSV offsets to PX4 North/East directly.  PX4 North is NOT the nose
    # unless the heading is 0, so rotate the file (x = ahead of nose, y = right) by the heading
    # recorded at ORIGIN_SET.  --traj-frame ned keeps the old behaviour.
    # The firmware plays the file relative to where PX4 says the drone is when START arrives.
    # After go_to_start the drone holds the aim point (0, 0 minus its settle offset), so the path is
    # sent relative to that same point from the first sample: no jump at the start, and the settle
    # offset is allowed for along the whole path.  Without a move: old 3 s blend back onto (0, 0).
    # The position is already the tag-20 one when tag 20 re-set the zero.
    aimed = False
    if (not getattr(args, "no_recentre", False) and state.get("start_moved") and cam.pipeline.origin is not None
            and link.pkt is not None):
        n0, e0 = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
        an, ae = state.get("start_aim") or (0.0, 0.0)
        if math.hypot(an - n0, ae - e0) <= 1.0:
            pts = pts.copy()
            pts[:, 0] += an - n0
            pts[:, 1] += ae - e0
            aimed = True
            rec.note(f"path sent from the aim point ({an:+.2f}, {ae:+.2f}); PX4 now ({n0:+.2f}, {e0:+.2f})")
    if not aimed and not getattr(args, "no_recentre", False) and cam.pipeline.origin is not None and link.pkt is not None:
        n0, e0 = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
        if math.hypot(n0, e0) <= 1.0:
            k = np.clip(np.arange(len(pts)) * f.dt_s / 3.0, 0.0, 1.0)
            if window is not None and window[0] > 0:
                # outward loop: from where the drone hovers after looking at tag 20, straight on to
                # the loop - the offset is taken out over the whole way out, not pulled back first
                k = np.clip(np.arange(len(pts)) / float(window[0]), 0.0, 1.0)
            pts = pts.copy()
            pts[:, 0] -= k * n0
            pts[:, 1] -= k * e0
            rec.note(f"after take-off the drone is at ({n0:+.2f}, {e0:+.2f}) m: "
                     + ("it flies from there straight on to the loop" if window is not None
                        else "path shifted so it begins on (0, 0)"))
        else:
            rec.note(f"take-off drift ({n0:+.2f}, {e0:+.2f}) m is over 1 m: NOT corrected")
    # --extend-right: stretch ONLY the right half of the path (y > 0) outwards so its widest point
    # moves out by this many metres.  The left half, the start and the end are untouched.
    if getattr(args, "extend_right", 0.0):
        pts = pts.copy()
        ymax = float(pts[:, 1].max())
        if ymax > 0.05:
            scale = (ymax + args.extend_right) / ymax
            right = pts[:, 1] > 0
            pts[right, 1] *= scale
            rec.note(f"right side stretched {args.extend_right:+.2f} m (widest right {ymax:.2f} -> "
                     f"{ymax * scale:.2f} m); left side unchanged")
    # Constant sideways lean (this drone drifts left): shift the path right by --shift-right m,
    # blended in over the first 3 s and back out over the last 3 s so it still lands on the start.
    if getattr(args, "shift_right", 0.0):
        n = len(pts)
        tt = np.arange(n) * f.dt_s
        T = tt[-1]
        k = np.clip(np.minimum(tt / 3.0, (T - tt) / 3.0), 0.0, 1.0)
        pts = pts.copy()
        pts[:, 1] += k * args.shift_right
        rec.note(f"path shifted {args.shift_right:+.2f} m to the right (lean compensation)")
    if getattr(args, "traj_frame", "nose") == "nose" and cam.pipeline.origin is not None:
        yaw0 = cam.pipeline.origin[2]
        cy, sy = math.cos(yaw0), math.sin(yaw0)
        pts = pts.copy()
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0], pts[:, 1] = x * cy - y * sy, x * sy + y * cy
        rec.note(f"trajectory rotated by heading {math.degrees(yaw0):+.0f} deg so +x = the nose at start")
    flags = 0
    if getattr(args, "yaw_follow", False) or window is not None:
        flags = 0x01
        rec.note("yaw-follow requested: the nose should turn along the path "
                 "(needs the yaw-follow ESP32 firmware; old firmware ignores it and keeps the heading)")
    yaw_limits = None
    if flags:
        import yaml as _yaml
        try:
            yc = (_yaml.safe_load(Path(args.config).read_text()) or {}).get("yaw") or {}
        except OSError:
            yc = {}
        yaw_limits = (float(yc.get("rate_max_dps", 30)), float(yc.get("accel_max_dps2", 30)))
        rec.note(f"nose turns at most {yaw_limits[0]:g} deg/s, rate building up at {yaw_limits[1]:g} deg/s^2 "
                 f"(config.yaml yaw:; needs the yaw v2 firmware - the first yaw patch ignores this and turns "
                 f"up to ~115 deg/s)")
    traj_id, t_start_sent, hover_xy = play_trajectory(args, comms, link, rec, pts, dt_ms, gtick, flags, window,
                                                      yaw_limits)
    state["t_traj_start"] = time.time() - (time.monotonic() - t_start_sent)
    rec.event("TRAJECTORY_STARTED", f"id {traj_id}, {duration:.1f} s")
    # the firmware plays pts as offsets from where it hovers when START arrives
    def expected_xy(t_mono: float):
        k = int(round((t_mono - t_start_sent) / f.dt_s))
        k = min(max(k, 0), len(pts) - 1)
        return hover_xy[0] + float(pts[k, 0]), hover_xy[1] + float(pts[k, 1])
    guard.start_trajectory(expected_xy)

    end = time.monotonic() + duration + 10.0
    next_pos = 0.0
    next_trk = 0.0
    while link.pkt.nav_state == NAV_TRAJ and time.monotonic() < end:
        if link.age() > f.stale_telemetry_s:
            raise ConnectionError(f"telemetry stale ({link.age():.1f} s)")
        gtick()
        pp = link.pkt
        if pp is not None and time.monotonic() >= next_trk:
            next_trk = time.monotonic() + 0.1
            tn, te = cam.pipeline._to_origin(pp.ned_x, pp.ned_y)
            state.setdefault("track", []).append((time.time() - (rec.t0 or time.time()), tn, te, pp.ned_x, pp.ned_y))
            state.setdefault("headings", []).append(pp.heading_rad)
        if time.monotonic() >= next_pos:
            next_pos = time.monotonic() + 1.0
            p = link.pkt
            sn, se = cam.pipeline._to_origin(p.ned_x, p.ned_y)
            conf = tag_map.confirmed_ids()
            cand = [f"{t}({tag_map.frame_count(t)}/{S.min_obs})" for t in tag_map.ids() if t not in conf]
            rec.note(f"pos from start ({sn:+.2f}, {se:+.2f})  confirmed {conf}  candidates {cand or '-'}")
        time.sleep(0.1)
    state["t_traj_end"] = time.time()
    rec.event("TRAJECTORY_ENDED", f"nav state {link.pkt.nav_state_name}")
    guard.stop()
    rec.note(guard.summary())
    if (getattr(args, "yaw_follow", False) or window is not None) and state.get("headings"):
        h = np.unwrap(np.array(state["headings"]))
        turned = math.degrees(float(h.max() - h.min()))
        if turned > 90:
            rec.note(f"yaw-follow: heading swept {turned:.0f} deg during the trajectory - working")
        else:
            rec.note(f"yaw-follow: heading only changed {turned:.0f} deg - the ESP32 firmware probably "
                     f"does not have the yaw-follow patch (flown at a fixed heading)")
    if window is not None and state.get("headings") and cam.pipeline.origin is not None:
        # outward loop: was the nose 90 deg right of the path (outward-yaw firmware) or along it (older)?
        try:
            exp = np.loadtxt(args.trajectory, delimiter=",", skiprows=1)[:, 4]     # planned nose (start frame)
            t_rel0 = state["t_traj_start"] - (rec.t0 or state["t_traj_start"])
            yaw0 = cam.pipeline.origin[2]
            d_out, d_old = [], []
            for (tt, *_), hd in zip(state["track"], state["headings"]):
                k = int(round((tt - t_rel0) / f.dt_s))
                if window[0] + 40 <= k < window[1]:              # past the first turn into the loop
                    e = (hd - yaw0 - exp[k] + math.pi) % (2 * math.pi) - math.pi
                    d_out.append(abs(e))
                    d_old.append(abs((e + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi))
            if d_out:
                mo, mold = float(np.median(d_out)), float(np.median(d_old))
                if mo < mold:
                    rec.note(f"outward nose: within {math.degrees(mo):.0f} deg of the planned outward heading - working")
                else:
                    rec.note(f"outward nose: the nose pointed ALONG the path in the loop ({math.degrees(mold):.0f} deg "
                             f"off that), not outward - flash the yaw v2 firmware patch (firmware_yaw_v2_patch.zip)")
        except (OSError, ValueError, IndexError, KeyError):
            pass

    # ---- return + land ------------------------------------------------------------------------
    # The trajectory ends at its first point (the takeoff point), so landing here IS the
    # return-to-start for this baseline test.  With --home-tag the drone first uses the HOME tag
    # to correct PX4's drift and land on the real start spot.
    if side.get("land_at_end") or getattr(args, "land_at_end", False):
        rec.note("landing straight away at the end of the path (no homing corrections)")
    elif args.home_tag is not None:
        if link.pkt.nav_state == NAV_ARRIVED:
            if home is None:
                home = HomeTag(args, S, cam, state["home_sightings"])
            return_home(args, S, comms, link, cam, rec, guard, gtick, home, state)
        else:
            rec.note(f"home: trajectory ended in nav state {link.pkt.nav_state_name}, not homing")
    state["t_land"] = time.time()
    state["land_sent"] = send_land(comms, args.drone_id, rec, "end of trajectory")
    state["rows"] = tag_table(rec, tag_map, refs, "PROVISIONAL", S.min_obs)
    sleep_watch(f.post_land_observe_s, link, f.stale_telemetry_s * 4)


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Search mission (--mission search): see search_mission.py
# ---------------------------------------------------------------------------------------
def _yaw_limits_from(args):
    import yaml as _yaml
    try:
        yc = (_yaml.safe_load(Path(args.config).read_text()) or {}).get("yaw") or {}
    except OSError:
        yc = {}
    return float(yc.get("rate_max_dps", 30)), float(yc.get("accel_max_dps2", 30))


def fly_pieces(args, S, comms, link, cam, rec, guard, gtick, state, P_abs, flags, window, label, z=None,
               blend_pts=None, tag_map=None, area=None, cuts=None, shift_fn=None, last_yaw_limits=None,
               follow=False):
    """5 Nov (3): a path longer than the drone takes in one go (TRAJ_MAX_PTS = 2400 points = 120 s) is
    flown as several pieces, one after the other (a short hover between them while the next one
    uploads).  Slower flights = more points, no firmware change.
    7 Nov 19:10: cuts = extra split points (the nose-in loop: after every corner + the closing U-turn);
    shift_fn() = (north, east) added to each piece before it is sent (the anchor fix); last_yaw_limits =
    nose rate for the LAST piece (the U-turn turns the nose faster than the search does)."""
    n, cap = len(P_abs), TRAJ_MAX_PTS - 40
    if cuts or shift_fn is not None or follow:
        bounds = sorted(set([0, n - 1] + [int(c) for c in (cuts or []) if 0 < int(c) < n - 1]))
        ranges = []
        for a0, b0 in zip(bounds[:-1], bounds[1:]):
            m_ = int(math.ceil((b0 - a0) / cap))
            st = int(math.ceil((b0 - a0) / m_))
            ranges += [(i0, min(b0, i0 + st)) for i0 in range(a0, b0, st)]
        rec.note(f"{label}: {n} points ({n * S.flight.dt_s:.0f} s) - flown in {len(ranges)} pieces "
                 f"{[round((i1 - i0) * S.flight.dt_s) for i0, i1 in ranges]} s")
        sent = state.setdefault("plan_sent_xy", np.array(P_abs, float).copy()) if label == "ellipse 1" else None
        for k, (i0, i1) in enumerate(ranges):
            sh = np.asarray(shift_fn(), float) if shift_fn is not None else np.zeros(2)
            Pk = np.asarray(P_abs[i0:i1 + 1], float) + sh
            if sent is not None:
                sent[i0:i1 + 1] = Pk
            state["last_piece_shift"] = sh
            w = None if window is None else (max(0, window[0] - i0), max(0, min(i1 + 1, window[1]) - i0))
            if w is not None and w[1] <= w[0]:
                w = None
            last = k == len(ranges) - 1
            fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Pk,
                        0x01 if follow else (flags if w is not None else 0x00), w, label if k == 0 else f"{label} (piece {k + 1}/{len(ranges)})",
                        z=None if z is None or k else z[i0:i1 + 1],
                        blend_pts=blend_pts if k == 0 else int(1.0 / S.flight.dt_s),
                        tag_map=tag_map, area=area, yaw_limits=last_yaw_limits if last else None)
        return
    if n <= cap:
        return fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, P_abs, flags, window, label,
                           z=z, blend_pts=blend_pts, tag_map=tag_map, area=area)
    parts = int(math.ceil(n / cap))
    size = int(math.ceil(n / parts))
    rec.note(f"{label}: {n} points ({n * S.flight.dt_s:.0f} s) - flown in {parts} pieces of <= {size}")
    for k, i0 in enumerate(range(0, n, size)):
        i1 = min(n, i0 + size + 1)                    # pieces share their end / start point
        w = None if window is None else (max(0, window[0] - i0), max(0, min(i1, window[1]) - i0))
        if w is not None and w[1] <= w[0]:
            w = None
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, P_abs[i0:i1],
                    flags if w is not None else 0x00, w, label if k == 0 else f"{label} (piece {k + 1}/{parts})",
                    z=None if z is None or k else z[i0:i1], blend_pts=blend_pts if k == 0 else int(1.0 / S.flight.dt_s),
                    tag_map=tag_map, area=area)


def fly_segment(args, S, comms, link: Link, cam, rec, guard, gtick, state: dict, P_abs, flags: int, window,
                label: str, z=None, blend_pts=None, tag_map=None, area=None, frame_yaw=None, yaw_limits=None):
    """Fly one planned piece.  P_abs: N x 2 in the (re-zeroed) start frame.  The firmware plays offsets
    from where PX4 says the drone is, so the piece starts at the drone and blends onto P_abs."""
    f = S.flight
    dt_ms = round(f.dt_s * 1000)
    p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
    n = len(P_abs)
    kb = blend_pts if blend_pts else min(n - 1, int(3.0 / f.dt_s))
    # 7 Nov (D26): piece 2 of the loop started 0.3 m from where the drone actually was and blended over 1 s
    # -> 0.31 m/s, over the cap, abort.  Blend at <= 0.10 m/s on top of the path, however far off it is.
    gap = float(np.hypot(*(P_abs[0] - p_now)))
    kb = int(min(n - 1, max(kb, math.ceil(gap / 0.10 / f.dt_s))))
    k = np.clip(np.arange(n) / max(1, kb), 0.0, 1.0)[:, None]
    off = (P_abs - P_abs[0]) + k * (P_abs[0] - p_now)
    pts = np.c_[off, np.zeros(n) if z is None else z]
    yaw0 = cam.pipeline.origin[2] if frame_yaw is None else frame_yaw
    cy, sy = math.cos(yaw0), math.sin(yaw0)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0], pts[:, 1] = x * cy - y * sy, x * sy + y * cy
    if len(pts) > TRAJ_MAX_PTS:
        raise RuntimeError(f"{label}: {len(pts)} points > {TRAJ_MAX_PTS}")
    vmax = float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).max() / f.dt_s)
    if vmax > f.max_speed_mps + 1e-6:
        raise RuntimeError(f"{label}: speed {vmax:.2f} m/s over the cap")
    loop = state.get("track_loop")
    if loop is not None:
        loop.start()                                  # no-op if already running / not supported
    _, t0, hover_xy = play_trajectory(args, comms, link, rec, pts, dt_ms, gtick, flags, window,
                                      yaw_limits or _yaw_limits_from(args))
    if loop is not None:
        loop.set_ref(t0, hover_xy, pts[:, :2], f.dt_s)
    rec.event("SEGMENT_STARTED", f"{label}: {n * f.dt_s:.1f} s", segment=label)
    if label == "ellipse 1":                          # the planned piece the deviation is measured on
        state["t_traj_start"] = time.time() - (time.monotonic() - t0)

    def expected_xy(t_mono: float, t0=t0, hover_xy=hover_xy, pts=pts):
        i = min(max(int(round((t_mono - t0) / f.dt_s)), 0), len(pts) - 1)
        return hover_xy[0] + float(pts[i, 0]), hover_xy[1] + float(pts[i, 1])
    guard.start_trajectory(expected_xy)
    end = time.monotonic() + n * f.dt_s + 10.0
    next_pos = next_trk = 0.0
    while link.pkt.nav_state == NAV_TRAJ and time.monotonic() < end:
        if link.age() > f.stale_telemetry_s:
            raise ConnectionError(f"telemetry stale ({link.age():.1f} s)")
        gtick()
        pp = link.pkt
        if pp is not None and time.monotonic() >= next_trk:
            next_trk = time.monotonic() + 0.1
            tn, te = cam.pipeline._to_origin(pp.ned_x, pp.ned_y)
            state.setdefault("track", []).append((time.time() - (rec.t0 or time.time()), tn, te, pp.ned_x, pp.ned_y))
            state.setdefault("headings", []).append(pp.heading_rad)
        if tag_map is not None and time.monotonic() >= next_pos:
            next_pos = time.monotonic() + 1.0
            sn, se = cam.pipeline._to_origin(pp.ned_x, pp.ned_y)
            tg = search_targets(tag_map, args, area)
            rec.note(f"{label}: pos ({sn:+.2f}, {se:+.2f})  targets found {[t for t, *_ in tg] or '-'}  "
                     f"all confirmed {tag_map.confirmed_ids()}")
        time.sleep(0.1)
    if loop is not None:
        loop.clear_ref()
    guard.stop()
    if link.pkt.nav_state != NAV_ARRIVED:
        raise RuntimeError(f"{label} ended in nav state {link.pkt.nav_state_name}")


def _track_loop_cfg(args) -> dict:
    """config.yaml track_loop: section (missing = defaults)."""
    import yaml
    try:
        return (yaml.safe_load(Path(args.config).read_text()) or {}).get("track_loop") or {}
    except (OSError, ValueError):
        return {}


def _tuning(args) -> dict:
    """config.yaml tuning: section (missing = defaults)."""
    import yaml
    try:
        return (yaml.safe_load(Path(args.config).read_text()) or {}).get("tuning") or {}
    except (OSError, ValueError):
        return {}


def wait_up(link, f, gtick, max_s: float, rec, up_m: float = 0.45, min_s: float = 4.0) -> None:
    """After CMD_START: move on as soon as PX4 says the drone is up (>= up_m for 1 s), at most max_s
    (5 Nov: the fixed 12 s wait was ~3 s longer than the climb)."""
    t0 = time.monotonic()
    up_since = None
    while time.monotonic() - t0 < max_s:
        sleep_watch(0.1, link, f.stale_telemetry_s, tick=gtick)
        z = getattr(link.pkt, "ned_z", None) if link.pkt else None
        if z is not None and -z >= up_m:
            up_since = up_since or time.monotonic()
            if time.monotonic() - up_since >= 1.0 and time.monotonic() - t0 >= min_s:
                rec.note(f"up at {-z:.2f} m after {time.monotonic() - t0:.1f} s")
                return
        else:
            up_since = None


def search_targets(tag_map, args, area):
    """Confirmed tags that count as targets: not the home tag or the anchor, inside the area (+0.4 m:
    a wrong decode usually lands somewhere odd).  Most-seen first."""
    out = []
    skip = {args.home_tag, int(args.anchor) if str(args.anchor).isdigit() else None,
            *(getattr(args, "extra_anchors", {}) or {})}
    for tid in tag_map.confirmed_ids():
        if tid in skip:
            continue
        e = tag_map.estimate(tid)
        if e is None:
            continue
        if area is not None and not (area.x[0] - 0.4 <= e.north <= area.x[1] + 0.4
                                     and area.y[0] - 0.4 <= e.east <= area.y[1] + 0.4):
            continue
        out.append((tid, e.north, e.east, tag_map.frame_count(tid)))
    return sorted(out, key=lambda r: -r[3])


class AnchorFix:
    """7 Nov 19:10: use the known anchors (tuning anchor_fix_ids, e.g. 7 at (2, -1) and 1 at (2, +1)) as a
    gauge of where the drone really is.  Every accepted sighting of an anchor puts it at PX4's position +
    what the camera measured; the difference to its taped spot is PX4's own error at that moment
    (F5-F8: 0.3-0.9 m, the same within ~5 cm over a whole pass).  So before each piece of the loop the
    newest clean pass (>= min_frames 0-bit frames in the last window_s) gives the shift: flying the
    path + that shift by PX4 puts the drone on the path ON THE FLOOR.  Capped at max_m; a pass saying
    more than reject_m is ignored as a bad read.  Without a new pass the last shift is kept."""

    def __init__(self, tag_map, area, tun, rec):
        self.tm, self.rec = tag_map, rec
        self.on = bool(tun.get("anchor_fix", False))
        ids = tun.get("anchor_fix_ids") or [7, 1]
        known = dict(getattr(area, "anchors", {}))
        known[int(area.home_id)] = tuple(area.home_at)
        self.known = {int(k): np.asarray(known[int(k)], float) for k in ids if int(k) in known}
        self.max_m = float(tun.get("anchor_fix_max_m", 0.4))
        self.reject_m = float(tun.get("anchor_fix_reject_m", 1.2))
        self.min_frames = int(tun.get("anchor_fix_min_frames", 8))
        self.window_s = float(tun.get("anchor_fix_window_s", 45.0))
        self.shift = np.zeros(2)
        self.used_t = 0.0                              # newest sighting already used
        self.history = []
        if self.on:
            rec.note(f"anchor fix ON: anchors {', '.join(f'{k} at ({v[0]:g}, {v[1]:+g})' for k, v in self.known.items())}"
                     f" shift the loop by PX4's error (<= {self.max_m:g} m)")

    def _obs(self, tid):
        with self.tm._lock:
            tr = self.tm._tracks.get(tid)
            return list(tr.obs) if tr else []

    def __call__(self):
        if not self.on:
            return self.shift
        now = time.time()
        best = None
        for tid, k in self.known.items():
            ob = [o for o in self._obs(tid) if o.t_epoch >= now - self.window_s and o.hamming == 0
                  and o.north is not None and (o.range_m or 9) <= 1.6]
            if len(ob) < self.min_frames:
                continue
            E = np.array([(o.north - k[0], o.east - k[1]) for o in ob])
            med = np.median(E, axis=0)
            spread = float(np.percentile(np.hypot(*(E - med).T), 80))
            t_last = max(o.t_epoch for o in ob)
            if spread > 0.15:
                self.rec.note(f"anchor fix: tag {tid} reads scattered ({spread:.2f} m) - not used")
                continue
            if best is None or t_last > best[0]:
                best = (t_last, tid, med, spread, len(ob))
        if best is None or best[0] <= self.used_t:
            if self.history:
                self.rec.note(f"anchor fix: no new anchor pass - keeping the shift ({self.shift[0]:+.2f}, {self.shift[1]:+.2f})")
            return self.shift
        t_last, tid, med, spread, nob = best
        self.used_t = t_last
        mag = float(np.hypot(*med))
        if mag > self.reject_m:
            self.rec.event("ANCHOR_FIX_REJECTED", f"tag {tid}: PX4 {mag:.2f} m off by it - more than "
                           f"{self.reject_m:g} m, taken as a bad read; shift stays ({self.shift[0]:+.2f}, {self.shift[1]:+.2f})")
            return self.shift
        new = med * min(1.0, self.max_m / mag) if mag > 1e-9 else med
        self.shift = new
        self.history.append((tid, float(med[0]), float(med[1])))
        self.rec.event("ANCHOR_FIX", f"tag {tid} ({nob} frames, spread {spread:.2f} m): PX4 is ({med[0]:+.2f}, {med[1]:+.2f}) m "
                       f"off -> next piece shifted ({new[0]:+.2f}, {new[1]:+.2f}) m", tag_id=tid,
                       px4_err_n=float(med[0]), px4_err_e=float(med[1]), shift_n=float(new[0]), shift_e=float(new[1]))
        return self.shift


def run_search(args, S, comms: CommsNode, link: Link, cam: CameraWorker, rec: Recorder,
               tag_map: TagMap, known: dict, state: dict) -> None:
    import search_mission as sm
    f = S.flight
    area = state["area"]
    cam_model = mt_camera = None
    rec.note(f"search area: x {area.x[0]:g}..{area.x[1]:g} m ahead, y {area.y[0]:+g}..{area.y[1]:+g} m; "
             f"{len(area.spots)} possible spots; find {area.targets} targets + home {args.home_tag} + at least "
             f"one map anchor of " + ", ".join(f"{k} at ({v[0]:g}, {v[1]:g})" for k, v in area.anchors.items()))
    # ---- correction loop (needs the track-loop firmware; off by itself on firmware without it)
    tl_cfg = _track_loop_cfg(args)
    if tl_cfg.get("enabled", True) and not args.no_track_loop:
        from miniproject.track_loop import TrackLoop
        loop = TrackLoop(args.esp_ip, args.cmd_port, link, rec, rec.out_dir,
                         kp=float(tl_cfg.get("kp", 0.3)), ki=float(tl_cfg.get("ki", 0.35)),
                         max_m=float(tl_cfg.get("max_m", 0.35)), lag_s=float(tl_cfg.get("lag_s", 0.4)),
                         rate_hz=float(tl_cfg.get("rate_hz", 5.0)))
        if loop.probe():
            state["track_loop"] = loop
            area.pull, area.expect_drift = 0.0, 0.0       # the loop removes the drift: send the plan as it is
            rec.note("correction loop ON: the drone answers CMD_TRAJ_NUDGE - it will be pulled back onto "
                     "the path (outward_pull_m not needed, set to 0 for this run)")
        else:
            loop.stop()
            rec.note("correction loop off: no answer to CMD_TRAJ_NUDGE (firmware without the track-loop "
                     "patch) - flying with the fixed outward_pull_m instead")
    print("planning the ellipse for this area (about 30 s) ...", flush=True)
    import make_trajectory as _mt
    mt_camera = _mt.camera_model(args.height)
    prm, Q1, w1, f1 = sm.plan_first(area, mt_camera)
    Y1 = sm.headings(Q1, w1)
    G1, seen1 = sm.field_seen(Q1, Y1, mt_camera, area)
    state["plan_xy"] = Q1.copy()
    state["plan_dt"] = f.dt_s
    T = (prm["xB"], prm["yc"])
    rec.note(f"ellipse 1: tip T ({T[0]:.2f}, {T[1]:+.2f}), {2 * prm['a']:.2f} m long x {2 * prm['b']:.2f} m wide, "
             f"{len(Q1) * f.dt_s:.0f} s; camera frames on each spot / anchor {f1}; "
             f"sees {100 * seen1.mean():.0f}% of the area")
    answer = input(f"Type TRAJ-{args.drone_id} to take off and fly the search mission: ").strip()
    if answer != f"TRAJ-{args.drone_id}":
        rec.note("confirmation did not match - nothing sent")
        return
    # ---- take off, home tag ------------------------------------------------------------------
    set_origin(cam, link, rec)
    guard, gtick = make_guard(S, cam, link, rec)
    state["guard"] = guard
    tag_map.reset()
    rec.set_t0()
    if not comms.send_command(args.drone_id, CommandPacket(CMD_START)):
        raise RuntimeError(f"CMD_START not sent: drone {args.drone_id} IP unknown")
    state["start_sent"] = True
    state["t_takeoff"] = time.time()
    rec.event("TAKEOFF_COMMAND_SENT", f"drone {args.drone_id}: CMD_START (arm + take off to 0.5 m)")
    tun = _tuning(args)
    wait_up(link, f, gtick, float(tun.get("takeoff_wait_max_s", f.takeoff_wait_s)), rec)
    home = HomeTag(args, S, cam, state["home_sightings"])
    state["home"] = home
    state["search_home"] = {"seen_takeoff": False, "seen_back": False}
    if home.start(link, gtick, rec, cam) is not None:
        rezero_from_home(cam, rec, tag_map, state, home, args.home_tag)
        state["home_drift_start"] = home.drift_start
        state["search_home"]["seen_takeoff"] = True
    # ---- 1: out (nose forward), ellipse (nose outward), back at T ----------------------------
    c1 = sm.pulled(prm, area.pull)                 # what is SENT (drifts back out onto Q1)
    Q1c, w1c = sm.segment1(c1["xB"], c1["yc"], c1["a"], c1["b"], area)
    if area.pull > 0:
        rec.note(f"sent ellipse pulled in by {prm['b'] - c1['b']:.2f} m on every side "
                 f"({2 * c1['a']:.2f} x {2 * c1['b']:.2f} m) - drone 5 drifts out the way its nose points")
    n1 = len(Q1c)
    z = np.zeros(n1)
    if args.height > 0.5:                             # climb over the first 6 s, stay up
        tt = np.arange(n1) * f.dt_s
        z = -(args.height - 0.5) * np.where(tt < 6.0, 0.5 - 0.5 * np.cos(np.pi * tt / 6.0), 1.0)
    state["t_traj_start"] = time.time()
    # 7 Nov 19:10: nose-in loop -> one piece per side + the closing U-turn (faster nose), anchor fix
    cuts1, u0 = sm.loop_cuts(Q1c, c1["xB"], c1["yc"], c1["a"], c1["b"], area)
    fix = AnchorFix(tag_map, area, tun, rec)
    state["anchor_fix"] = fix
    tdps_u = float(tun.get("turn_home_dps", 20.0))
    lanes = sm.is_lanes(area)                         # 7 Nov 21:30: nose follows the path the whole way
    fly_pieces(args, S, comms, link, cam, rec, guard, gtick, state, Q1c, 0x01, None if lanes else w1c,
               "ellipse 1", z=z, blend_pts=int(3.0 / f.dt_s) if lanes else w1c[0], tag_map=tag_map, area=area,
               cuts=cuts1 or None, shift_fn=fix if fix.on else None,
               last_yaw_limits=(tdps_u, tdps_u) if u0 is not None else None, follow=lanes)
    state["ended_facing_pad"] = u0 is not None or lanes
    state["t_traj_end"] = time.time()
    found = search_targets(tag_map, args, area)
    anchors_seen = [k for k in area.anchors if tag_map.is_confirmed(int(k))]
    rec.event("SEARCH_CHECK", f"after ellipse 1: {len(found)}/{area.targets} targets "
              f"{[t for t, *_ in found]}, map anchors seen {anchors_seen or 'NONE'} "
              f"(of {list(area.anchors)}), home {args.home_tag} "
              f"{'seen' if state['search_home']['seen_takeoff'] else 'NOT seen'}",
              found=[t for t, *_ in found], anchors_seen=anchors_seen)
    # ---- 2: adjusted ellipses while targets are missing --------------------------------------
    seen = seen1
    for r in range(args.search_retries):
        # 5 Nov rule: backup ONLY if the targets are not exactly right or NO map anchor (5/11/12) was
        # seen.  One anchor + tag 20 is enough; never fly again just to collect more anchors.
        # 5 Nov (2): AT MOST area.targets targets.  4+ confirmed (D7: a wrong decode) is NOT a reason to
        # fly again - the most-seen 3 are kept (search_targets sorts by frames), the rest are listed
        # as "probably not real".  Backup only if FEWER than 3 targets or no map anchor at all.
        anchor_ok = any(tag_map.is_confirmed(int(k)) for k in area.anchors)
        if len(found) > area.targets:
            rec.note(f"{len(found)} targets confirmed, at most {area.targets}: keeping the {area.targets} most seen "
                     f"{[t for t, *_ in found[:area.targets]]}, ignoring {[t for t, *_ in found[area.targets:]]}")
        if len(found) >= area.targets and anchor_ok:
            break
        why = []
        if len(found) < area.targets:
            why.append(f"{len(found)} targets, not {area.targets}")
        if not anchor_ok:
            why.append(f"no map anchor ({', '.join(map(str, area.anchors))}) seen")
        rec.note(f"backup needed: {'; '.join(why)}")
        print(f"planning backup ellipse {r + 1} ...", flush=True)
        res = sm.plan_retry(area, prm, [(n, e) for _, n, e, _ in found], mt_camera, seen,
                            anchor_missing=not anchor_ok)
        if res is None:
            rec.note("no backup ellipse fits - going home")
            break
        prm2, Q2, w2, left = res
        rec.event("SEARCH_MORE", f"{'; '.join(why)}: backup ellipse {r + 1}, "
                  f"{2 * prm2['a']:.2f} x {2 * prm2['b']:.2f} m from tip ({prm2['xB']:.2f}, {prm2['yc']:+.2f}), {len(Q2) * f.dt_s:.0f} s"
                  + (f", aimed at spots {[tuple(round(v, 2) for v in s) for s in left]}" if left else
                     ", aimed at the ground not seen yet"))
        c2 = sm.pulled(prm2, area.pull)
        Q2c, w2c = sm.segment2(c2["xB"], c2["yc"], c2["a"], c2["b"], area)
        p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        if math.hypot(c2["xB"] - p_now[0], c2["yc"] - p_now[1]) > 0.05:     # backup has its own (centred) tip
            Qm = sm.segment3(tuple(p_now), (c2["xB"], c2["yc"]), speed=float(tun.get("lead_speed_mps", 0.18)))
            fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qm, 0x00, None,
                        f"to the backup tip ({c2['xB']:.2f}, {c2['yc']:+.2f})")
        fly_pieces(args, S, comms, link, cam, rec, guard, gtick, state, Q2c, 0x01, w2c, f"backup ellipse {r + 1}",
                   tag_map=tag_map, area=area)
        state["ended_facing_pad"] = False
        state["backup_flown"] = True
        Y2 = sm.headings(Q2, w2, h0=math.pi)
        seen = seen | sm.field_seen(Q2, Y2, mt_camera, area)[1]
        found = search_targets(tag_map, args, area)
        rec.event("SEARCH_CHECK", f"after backup ellipse {r + 1}: {len(found)}/{area.targets} targets "
                  f"{[t for t, *_ in found]}", found=[t for t, *_ in found])
    # ---- 3: back towards the pad (nose held on it), look at tag 20, onto the pad, land -------
    L = sm.look_point(area, *T)
    p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
    ret_speed = float(tun.get("return_speed_mps", 0.2))
    look_s = float(tun.get("home_look_s", args.home_look_s))
    if state.get("ended_facing_pad"):
        # 7 Nov 19:10: the loop closed with the U-turn: the drone is ON the look point, nose on the pad.
        # Stay where it is (the anchor fix's shift included) - no 'back to the look point' move.
        L = L + np.asarray(state.get("last_piece_shift", np.zeros(2)), float)
        rec.note(f"loop ended on the look point facing the pad: at ({p_now[0]:+.2f}, {p_now[1]:+.2f}) by PX4, "
                 f"look point + anchor shift ({L[0]:+.2f}, {L[1]:+.2f})")
    elif (getattr(area, "main_nose", "out") == "in" and not getattr(args, "blind_home", False)
          and not state.get("backup_flown")):
        # 7 Nov: after a nose-IN loop the nose points AWAY from the pad.  Go just past the look point,
        # then come back onto it with the nose following the direction of travel (yaw-follow, no window)
        # -> it ends facing the pad, as the tag-23 look and the landing expect.  The turn may go faster
        # than the search's slow nose (tuning turn_home_dps), the drone is only hovering.
        hxy = np.asarray(area.home_at, float)
        u = (L - hxy) / max(1e-6, float(np.linalg.norm(L - hxy)))
        A = L + 0.45 * u
        Qa = sm.segment3(tuple(p_now), tuple(A), speed=ret_speed)
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qa, 0x00, None, "past the look point")
        tdps = float(tun.get("turn_home_dps", 20.0))
        # keep moving towards the pad (slowly, 0.45 m) for the WHOLE turn: the nose follows the direction
        # of travel, so it must not stop moving before the nose has come round (sim: it turned back).
        n_m = int((180.0 / tdps + 4.0) / f.dt_s)
        Qt = np.r_[np.linspace(A, L, n_m), np.repeat(np.asarray(L, float)[None, :], int(0.5 / f.dt_s), axis=0)]
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qt, 0x01, None,
                    f"turn to face the pad ({tdps:.0f} deg/s)", blend_pts=1, yaw_limits=(tdps, tdps))
        p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
    # 5 Nov experiment (tuning blind_home / --blind-home): fly straight back to (0, 0) as the drone
    # thinks it is (the zero set from tag 20 right after take-off), NO look at tag 20, NO drift
    # correction, NO pad re-aim, NO land offsets.  Then the same slow descent and LAND.
    blind = bool(getattr(args, "blind_home", False) or tun.get("blind_home", False))
    if blind:
        p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        rec.event("HOME_BLIND", f"blind return: from ({p_now[0]:+.2f}, {p_now[1]:+.2f}) straight to (0, 0) by PX4 only "
                  f"(no tag-20 look, no drift correction, no pad re-aim)", px4_x=float(p_now[0]), px4_y=float(p_now[1]))
        # tuning land_offset_*: drone 5 slides LEFT in the last metre down (take-off: tag 20 shows it 0.15-0.55 m
        # left after every take-off on 5 Nov), so aim that much right of the pad centre
        hx = float(tun.get("land_offset_ahead_m", 0.0) or 0.0)
        hy = float(tun.get("land_offset_right_m", 0.0) or 0.0)
        Qb = sm.segment3(tuple(p_now), (hx, hy), speed=ret_speed)
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qb, 0x00, None, "blind back to (0, 0)")
        n0, e0, _ln, _le = settled_position(link, cam, f, gtick, settle_s=0.8, samples=2)
        rec.note(f"blind return: drone thinks it is at ({n0:+.2f}, {e0:+.2f}) - TAPE-MEASURE where it really lands")
    else:
        if math.hypot(*(L - p_now)) > 0.05:
            Q3 = sm.segment3(tuple(p_now), tuple(L), speed=ret_speed)
            fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Q3, 0x00, None, "back to the look point")
        obs = home.look(link, gtick, look_s, t_from=time.monotonic())
        if len(obs) < 3 and area.look_dist > 0.75:
            # D1/D2 (3 Nov): tag 20 never read at the look point (glare there, and drone 5 stops ~0.25 m
            # to its left).  Move 0.15 m closer, aiming past by what the first stop missed, look again.
            p_now = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
            hxy = np.asarray(area.home_at, float)
            u = (L - hxy) / max(1e-6, float(np.linalg.norm(L - hxy)))
            L2 = hxy + u * (area.look_dist - 0.15)
            aim = L2 + np.array(_clamp(tuple(-args.aim_gain * (p_now - L)), args.aim_max))
            rec.note(f"tag {args.home_tag} not read from ({p_now[0]:+.2f}, {p_now[1]:+.2f}) - moving 0.15 m closer "
                     f"and looking again")
            if math.hypot(*(aim - p_now)) > 0.03:
                Q3b = sm.segment3(tuple(p_now), tuple(aim), speed=ret_speed)
                fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Q3b, 0x00, None, "closer look")
            obs = home.look(link, gtick, look_s, t_from=time.monotonic())
        hx, hy = (float(v) for v in args.home_at.split(","))
        # tuning land_offset_*: where to land relative to the pad centre (it lands LEFT -> aim right)
        hx += float(tun.get("land_offset_ahead_m", 0.0) or 0.0)
        hy += float(tun.get("land_offset_right_m", 0.0) or 0.0)
        if tun.get("land_target_m") is not None:          # 6 Nov: fly to THIS point (e.g. over tag 20) instead
            hx, hy = (float(v) for v in tun["land_target_m"])
        dn_end, de_end = 0.0, 0.0                         # PX4 minus true, back home (from tag 20)
        if len(obs) >= 3:
            h_true, h_err = home.heading_now(obs, link)
            fix = home.fix(obs, h_true)
            px = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
            state["home_drift_end"] = home.drift_end = (time.time(), px[0] - fix.drone_x, px[1] - fix.drone_y)
            dn_end, de_end = px[0] - fix.drone_x, px[1] - fix.drone_y
            state["search_home"]["seen_back"] = True
            rec.event("HOME_TAG_SEEN", f"on the way back: drone at ({fix.drone_x:+.2f}, {fix.drone_y:+.2f}) by tag "
                      f"{args.home_tag}, PX4 thinks ({px[0]:+.2f}, {px[1]:+.2f}); flying onto the pad",
                      drone_x=fix.drone_x, drone_y=fix.drone_y, px4_x=px[0], px4_y=px[1], heading_error_deg=h_err)
            dx, dy = hx - fix.drone_x, hy - fix.drone_y
            frame_yaw = link.pkt.heading_rad - h_true      # PX4's axes, corrected for its heading error
        else:
            px = cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y)
            rec.event("HOME_TAG_NOT_SEEN", f"on the way back: tag {args.home_tag} not seen - landing by PX4's (0, 0)")
            dx, dy = hx - px[0], hy - px[1]
            frame_yaw = None
        import make_trajectory as _mt2
        line = np.linspace((0.0, 0.0), (dx, dy), max(2, int(math.hypot(dx, dy) / 0.005)))
        Qf = _mt2.slow_turns(line, turn_dps=25.0, speed=ret_speed, end_hold=0.3,
                             start_heading=math.atan2(dy, dx))
        here = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qf + here, 0x00, None, "onto the pad",
                    blend_pts=1, frame_yaw=frame_yaw)
        # ---- settle check over the pad: drone 5 stops 0.15-0.25 m past where it is sent (D1, D2), so
        # measure where it settled (PX4, corrected by tag 20 when it was read) and re-aim past the miss
        goal = np.array([hx + dn_end, hy + de_end])       # the pad centre in PX4's numbers
        aim = goal.copy()
        pad_tol = float(tun.get("pad_tolerance_m", args.pad_tolerance))
        for k in range(int(tun.get("pad_tries", args.pad_tries))):
            n0, e0, ln, le = settled_position(link, cam, f, gtick, settle_s=0.8, samples=2)
            miss = np.array([n0, e0]) - goal
            rec.note(f"over the pad: {math.hypot(*miss):.2f} m from its centre ({miss[0]:+.2f} ahead, {miss[1]:+.2f} right)")
            if math.hypot(*miss) <= pad_tol:
                break
            aim = goal + np.array(_clamp(tuple(aim - goal - args.aim_gain * miss), args.aim_max))
            d = aim - np.array([ln, le])
            line = np.linspace((0.0, 0.0), tuple(d), max(2, int(math.hypot(*d) / 0.005)))
            Qc = _mt2.slow_turns(line, turn_dps=25.0, speed=0.15, end_hold=0.5, start_heading=math.atan2(d[1], d[0]))
            fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qc + np.array([ln, le]), 0x00, None,
                        f"pad correction {k + 1}", blend_pts=1)
    # 5 Nov: descend under position hold (the drone keeps holding the pad point; PX4's LAND from
    # 0.8 m takes ~8 s at 0.1 m/s and drifts left on the way down), then LAND from just above the pad
    low = float(tun.get("land_descend_to_m", 0.0) or 0.0)
    if low >= 0.03 and args.height - low > 0.05:          # 6 Nov: down to 5 cm allowed
        drop = args.height - low
        v = max(0.05, float(tun.get("land_descend_speed_mps", 0.25)))
        n_d = max(2, int(drop / v / f.dt_s))
        p0 = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        Qd = np.repeat(p0[None, :], n_d + int(0.6 / f.dt_s), axis=0)
        zd = np.r_[np.linspace(0.0, drop, n_d), np.full(len(Qd) - n_d, drop)]   # + = down
        fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qd, 0x00, None,
                    f"descend to {low:.2f} m", z=zd, blend_pts=1)
    # 6 Nov 23:56: at the low height, LOOK at tag 20 again (it is just ahead of the nose: the drone stops
    # land_target_m short of it), then move land_creep_m forward + land_creep_right_m right of the nose,
    # measured FROM TAG 20 (0.75 forward of tag 20 = the pad centre), and LAND there.
    creep = float(tun.get("land_creep_m", 0.0) or 0.0)
    creep_r = float(tun.get("land_creep_right_m", 0.0) or 0.0)
    if abs(creep) > 0.005 or abs(creep_r) > 0.005:
        p0 = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        hn = link.pkt.heading_rad - cam.pipeline.origin[2]          # nose, in the start frame (PX4)
        fwd = np.array([math.cos(hn), math.sin(hn)])
        rgt = np.array([math.cos(hn + math.pi / 2), math.sin(hn + math.pi / 2)])
        tag_xy = np.array([float(v) for v in args.home_tag_at.split(",")])
        low_s = float(tun.get("land_low_look_s", 4.0) or 0.0)
        obs = home.look(link, gtick, low_s, need=5, t_from=time.monotonic()) if low_s > 0 else []
        # 7 Nov (D23): PX4's position JUMPED 0.67 m sideways while hovering at 0.15 m (flow / range at the
        # bottom of their range) and the next piece started from the old position -> 14 m/s, abort.
        # Re-read where PX4 is now; if it jumped during the look, do not fly a metre on it: land here.
        p_after = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
        jump = float(np.hypot(*(p_after - p0)))
        p0 = p_after
        fix = None
        if len(obs) >= 3:
            h_true, _he = home.heading_now(obs, link)
            fix = home.fix(obs, h_true)
        if fix is not None:
            px = np.array(cam.pipeline._to_origin(link.pkt.ned_x, link.pkt.ned_y))
            drift = px - np.array([fix.drone_x, fix.drone_y])          # PX4 minus true, right now
            goal_true = tag_xy + creep * fwd + creep_r * rgt
            p1 = goal_true + drift                                      # the same spot in PX4's numbers
            rec.event("HOME_TAG_LOW", f"at {low:.2f} m: tag {args.home_tag} read {len(obs)}x - drone at "
                      f"({fix.drone_x:+.2f}, {fix.drone_y:+.2f}), PX4 thinks ({px[0]:+.2f}, {px[1]:+.2f}); "
                      f"going to ({goal_true[0]:+.2f}, {goal_true[1]:+.2f}) = {creep:.2f} m forward + "
                      f"{creep_r:.2f} m right of tag {args.home_tag}",
                      drone_x=fix.drone_x, drone_y=fix.drone_y, goal_x=float(goal_true[0]), goal_y=float(goal_true[1]))
        else:
            # not read at the low height: same goal, from tag 20's spot in PX4's numbers (+ the drift
            # measured on the way back, if any)
            p1 = tag_xy + np.array([dn_end, de_end]) + creep * fwd + creep_r * rgt if not blind \
                else p0 + creep * fwd + creep_r * rgt
            rec.event("HOME_TAG_LOW_NOT_SEEN", f"at {low:.2f} m: tag {args.home_tag} not read ({len(obs)} sightings) - "
                      f"going {creep:.2f} m forward + {creep_r:.2f} m right of its spot by PX4")
        dist = float(np.hypot(*(p1 - p0)))
        max_jump = float(tun.get("land_max_jump_m", 0.25))
        if jump > max_jump and fix is None:
            rec.event("HOME_LOW_JUMP", f"PX4's position moved {jump:.2f} m while hovering at {low:.2f} m (its flow / "
                      f"range sensor at low height) - not flying {dist:.2f} m on it, landing HERE")
        elif dist > 1.6:
            rec.event("HOME_LOW_FAR", f"landing spot {dist:.2f} m away - too far to trust, landing HERE")
        else:
            n_c = max(2, int(dist / float(tun.get("land_creep_speed_mps", 0.08)) / f.dt_s))
            Qc2 = np.r_[np.linspace(p0, p1, n_c), np.repeat(p1[None, :], int(0.8 / f.dt_s), axis=0)]
            fly_segment(args, S, comms, link, cam, rec, guard, gtick, state, Qc2, 0x00, None,
                        f"to the landing spot at {low:.2f} m ({dist:.2f} m)", blend_pts=int(1.0 / f.dt_s))
    rec.event("HOME_REACHED", f"over the pad ({hx:+.2f}, {hy:+.2f}) - landing")
    state["t_land"] = time.time()
    state["land_sent"] = send_land(comms, args.drone_id, rec, "search mission done")
    if state.get("track_loop") is not None:
        state["track_loop"].stop()
    state["rows"] = tag_table(rec, tag_map, known, "PROVISIONAL", S.min_obs)
    state["search_found"] = search_targets(tag_map, args, area)
    sleep_watch(f.post_land_observe_s, link, f.stale_telemetry_s * 4)


def print_search_summary(out_dir: Path, args, area, opt_rc=None) -> None:
    """The big end-of-run list: home tag, anchor and the targets, with their positions.
    A target shows its optimised position only if the optimisation finished (opt_rc 0, or 2 =
    stopped at its iteration limit) AND the result is believable: inside the area and within
    0.5 m of the flight's own estimate.  Otherwise the flight's own (live) estimate is shown."""
    import csv as _csv
    import json as _json
    try:
        s = _json.loads((out_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return
    rows = s.get("tags", {})
    found = s.get("search_found") or []
    if not found:                       # run ended early (abort, link drop): rebuild it from the tags
        skip = {args.home_tag, int(args.anchor) if str(args.anchor).isdigit() else None,
                *(getattr(args, "extra_anchors", {}) or {})}
        found = sorted(((int(k), v["north_m"], v["east_m"], v.get("accepted_frame_count", v.get("n_obs", 0)))
                        for k, v in rows.items()
                        if v.get("confirmed") and int(k) not in skip
                        and area.x[0] - 0.4 <= v["north_m"] <= area.x[1] + 0.4
                        and area.y[0] - 0.4 <= v["east_m"] <= area.y[1] + 0.4),
                       key=lambda t: -t[3])
    opt = {}
    rp = out_dir / "optimization" / "tags.csv"            # every mapped tag, after the optimisation
    if opt_rc in (0, 2) and rp.exists():                  # a crashed solve's numbers are not used
        for r in _csv.DictReader(rp.open()):
            try:
                opt[int(r["tag_id"])] = (float(r["x_m"]), float(r["y_m"]))
            except (KeyError, ValueError):
                pass
    # 7 Nov (D28): the whole live map came out turned ~22 deg and ~10 % too big (drone not lined up at
    # take-off; tag 5 read 1.35 m right of its tape).  Fit ONE turn + scale + shift that puts the anchors
    # seen (5, 11, 12) and tag 20 back on their tape spots, and apply it to the targets ("anchor-fitted").
    fitT = None
    try:
        tapes = {int(k): tuple(map(float, v)) for k, v in (getattr(area, "anchors", {}) or {}).items()}
        tapes[int(args.home_tag)] = tuple(float(v) for v in args.home_tag_at.split(","))
        pairs = []
        for tid, tp in tapes.items():
            r = rows.get(str(tid))
            if r and r.get("confirmed") and "north_m" in r:
                pairs.append(((r["north_m"], r["east_m"]), tp))
            elif tid == int(args.home_tag):
                pairs.append((tp, tp))          # tag 20 re-set (0, 0) after take-off: there by construction
        if len(pairs) >= 3:
            import numpy as _np
            A = _np.array([p for p, _ in pairs], float); B = _np.array([q for _, q in pairs], float)
            ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
            U, Sv, Vt = _np.linalg.svd(a.T @ b)
            R = (U @ Vt).T
            if _np.linalg.det(R) < 0:
                Vt[-1] *= -1; R = (U @ Vt).T
            sc = float(Sv.sum() / max(1e-9, (a ** 2).sum()))
            sc = min(1.2, max(0.8, sc))
            res = _np.hypot(*((sc * (R @ a.T)).T + mb - B).T)
            ang = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            if float(res.max()) < 0.2:
                fitT = (sc, R, ma, mb)
                print(f"anchor fit: the live map is turned {-ang:+.1f} deg and {100 * (1 / sc - 1):+.0f} % off in size "
                      f"(from {len(pairs)} known tags, worst leftover {res.max():.2f} m) - targets corrected with it")
            else:
                print(f"anchor fit not used: the known tags do not agree (worst leftover {res.max():.2f} m)")
    except Exception as exc:                      # noqa: BLE001 - the list must still print
        print(f"anchor fit skipped: {exc}")

    def fitted(p):
        if fitT is None:
            return None
        sc, R, ma, mb = fitT
        import numpy as _np
        q = sc * (R @ (_np.asarray(p, float) - ma)) + mb
        return float(q[0]), float(q[1])
    lines = []

    def line(kind, tid, tape=None):
        r = rows.get(str(tid))
        if not r or not r.get("confirmed"):
            lines.append(f"  {kind:7s} {tid:>3}   NOT FOUND")
            return 0
        live = (r["north_m"], r["east_m"])
        pos, src = live, "live"
        o = opt.get(tid) if tape is None else None      # anchors are pinned to the tape: show live
        nfr = int(r.get('accepted_frame_count', r.get('n_obs', 0)) or 0)
        if o is not None:
            # 5 Nov (3): the optimised position only for a tag seen in >= opt_min_frames (30) frames (fewer:
            # too little for the solver - D15 tags 13 / 28 with 8-9 frames came out 1-3.5 m off) and inside
            # the area; otherwise the flight's own reading
            gap = math.hypot(o[0] - live[0], o[1] - live[1])
            inside = (area.x[0] - 0.4 <= o[0] <= area.x[1] + 0.4 and area.y[0] - 0.4 <= o[1] <= area.y[1] + 0.4)
            need = int(getattr(area, "opt_min_frames", 30))
            if nfr >= need and inside:
                pos, src = o, f"optimised ({gap:.2f} m from live)"
            elif nfr < need:
                src = f"live: only {nfr} frames (< {need}), optimised ({o[0]:.2f}, {o[1]:+.2f}) not trusted"
            else:
                src = f"live; optimised was {gap:.1f} m away, outside the area"
        if tape is None and src.startswith("live") and fitted(live) is not None:
            pos, src = fitted(live), "anchor-fitted (" + ("" if src == "live" else src + ", ") + \
                f"live ({live[0]:.2f}, {live[1]:+.2f}))"
        t = f"   (tape {tape[0]:.2f}, {tape[1]:+.2f})" if tape else ""
        lines.append(f"  {kind:7s} {tid:>3}   at ({pos[0]:.2f} ahead, {pos[1]:+.2f} right) m [{src}], "
                     f"{r.get('accepted_frame_count', r.get('n_obs', 0))} frames{t}")
        return 1
    hs = s.get("search_home") or {}
    tape = tuple(float(v) for v in args.home_tag_at.split(","))
    if hs.get("seen_takeoff") or hs.get("seen_back"):
        when = " and ".join(w for w, k in (("after take-off", "seen_takeoff"), ("on the way back", "seen_back"))
                            if hs.get(k))
        hr = rows.get(str(args.home_tag)) or {}
        pos = (hr["north_m"], hr["east_m"]) if "north_m" in hr else None
        where = f"measured at ({pos[0]:.2f}, {pos[1]:+.2f}) m, " if pos else ""
        lines.append(f"  {'home':7s} {args.home_tag:>3}   seen {when}; {where}tape ({tape[0]:.2f}, {tape[1]:+.2f})")
        n = 1
    else:
        lines.append(f"  {'home':7s} {args.home_tag:>3}   NOT SEEN")
        n = 0
    home_ok = n == 1
    n_anc = line("anchor", int(args.anchor) if str(args.anchor).isdigit() else -1, area.anchor_at)
    for _aid, _at in (getattr(args, "extra_anchors", {}) or {}).items():
        n_anc += line("anchor", int(_aid), _at)
    n += n_anc
    n_tg = 0
    for i, (tid, *_r) in enumerate(found[:area.targets]):
        n_tg += line(f"target{i + 1}", tid)
    n += n_tg
    for i in range(len(found), area.targets):
        lines.append(f"  target{i + 1}  ---   NOT FOUND")
    extra = [t for t, *_ in found[area.targets:]]
    total = 1 + area.targets + 1 + len(getattr(args, "extra_anchors", {}) or {})
    bar = "=" * 78
    need = home_ok and n_tg == area.targets and n_anc >= 1
    print(f"\n{bar}\nTAGS FOUND: {n}/{total}" + ("   ALL FOUND" if n == total else "")
          + f"   (needed: home {args.home_tag} + {area.targets} targets + >= 1 map anchor: "
          + ("YES" if need else "NO") + f")\n{bar}")
    print("\n".join(lines))
    if extra:
        print(f"  (also confirmed, fewer frames, probably not real targets: {extra})")
    print(bar)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone-id", type=int, required=True)
    ap.add_argument("--esp-ip", required=True, help="drone IP, 192.168.1.(200 + drone id)")
    ap.add_argument("--mode", choices=["observe", "fly"], required=True)
    ap.add_argument("--trajectory", nargs="+", metavar="CSV | SHAPE LENGTH WIDTH",
                    help="fly mode: a t,x,y,z CSV (e.g. experiment_plans/E1_ellipse.csv), or a shape and the box "
                         "size in metres, made on the spot by make_trajectory.py: ellipse 5 2, rectangle 5 2, "
                         "lawnmower_vertical 5 2, lawnmower_horizontal 5 2 (length = ahead, width = across)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--refs", help="CSV of reference tag positions: id,north_m,east_m")
    ap.add_argument("--no-optimize", action="store_true",
                    help="fly mode: skip the pose-graph optimisation that otherwise runs after landing")
    ap.add_argument("--anchor", default=None,
                    help="tag the optimisation pins the map to (its tape position must be known).  Default: "
                         "anchors.map_tag in config.yaml (28).  'auto' = the farthest truth tag seen")
    ap.add_argument("--anchor-at", default=None, metavar="X,Y",
                    help="tape position of the --anchor tag from the start circle: ahead of the nose, right (m).  "
                         "Default: anchors.map_tag_at in config.yaml (3.0,0), or its line in --refs")
    ap.add_argument("--optimize-timeout", type=float, default=1500.0, help="seconds (5 Nov: 600 -> 1500)")
    ap.add_argument("--expect-ids", help="tag IDs you expect to see, e.g. 3,5,13 (missing ones are logged as N)")
    ap.add_argument("--duration", type=float, default=0, help="observe mode: stop after this many seconds (0 = Ctrl+C)")
    ap.add_argument("--trial", default=1, help="worksheet: trial number")
    ap.add_argument("--pass-no", default=1, help="worksheet: pass number")
    ap.add_argument("--height", type=float, default=0.5,
                    help="flight height (m), 0.5-1.1.  The firmware takes off to 0.5 m; anything higher is "
                         "added to the trajectory (6 s climb at the start, 6 s descent at the end) and the "
                         "tag range limit grows with it.  Also written to the worksheet.")
    ap.add_argument("--speed", default="", help="worksheet: speed (m/s)")
    ap.add_argument("--mavlink-port", type=int, default=14550,
                    help="also read PX4 MAVLink here and fill z/quaternion/velocity/rates (0 = off)")
    ap.add_argument("--yaw-follow", action="store_true",
                    help="turn the nose along the path (heading tangent); needs the yaw-follow ESP32 firmware")
    ap.add_argument("--extend-right", type=float, default=0.0,
                    help="push only the RIGHT half of the path out by this many metres (left half unchanged)")
    ap.add_argument("--shift-right", type=float, default=0.0,
                    help="shift the whole path this many metres to the right (negative = left) to cancel "
                         "a constant sideways lean; blended in/out over 3 s, still starts and ends on the mark")
    ap.add_argument("--no-recentre", action="store_true",
                    help="do not shift the path back onto the start mark after take-off drift")
    ap.add_argument("--traj-frame", choices=["nose", "ned"], default="nose",
                    help="nose (default): CSV x = ahead of the nose at start, y = right; ned: raw PX4 North/East")
    ap.add_argument("--show", action="store_true",
                    help="live camera window with the tags outlined (green confirmed, yellow k/3 candidate, "
                         "red rejected).  In the window: q closes it, l = LAND (same as Ctrl+C)")
    ap.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True,
                    help="save the camera stream with the same overlays as <run>/camera.mp4 (5 Nov: ON by "
                         "default; --no-record-video turns it off)")
    ap.add_argument("--mount-config", default=str(default_mount_config()),
                    help="camera mount (lever arm + pitch) from this mapper config, so the live tag positions "
                         "and the optimisation use the same mount.  Default: mapper_config_current.yaml "
                         "(made by set_camera_mount.py) if it exists, else mapper_config_29_Oct.yaml; "
                         "'config' = use miniproject/config.yaml")
    ap.add_argument("--pitch", type=float, default=None, metavar="DEG",
                    help="camera tilt for this run, degrees nose-down (overrides the mount file), e.g. --pitch 45")
    ap.add_argument("--cam-fwd", type=float, default=None, metavar="M",
                    help="camera distance ahead of the PX4 for this run, metres, e.g. --cam-fwd 0.075")
    ap.add_argument("--cam-down", type=float, default=None, metavar="M",
                    help="camera distance below the PX4 for this run, metres, e.g. --cam-down 0.025")
    ap.add_argument("--sensor", action="append", default=[], metavar="NAME=VALUE",
                    help="camera chip setting for this run, e.g. --sensor ae_level=-2 --sensor brightness=0 "
                         "(needs the camera-settings firmware; see camera_settings.py for the names)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="change any config.yaml value for this run only, e.g. --set gate.max_range_m=1.5 "
                         "--set estimate.min_obs=2 (repeat for more).  config.yaml itself is not edited")
    ap.add_argument("--home-tag", default=None, metavar="ID",
                    help="take-off / landing tag (NOT one of the targets): re-sets (0, 0) after take-off, measures "
                         "PX4's drift and guides the landing.  Default: anchors.home_tag in config.yaml (20); "
                         "'none' = no home tag")
    ap.add_argument("--home-tag-at", default=None, metavar="X,Y",
                    help="home tag position from the start circle: ahead of the nose, right (m), tape-measured.  "
                         "Default: anchors.home_tag_at in config.yaml (0.75,0 for the 23 deg camera at 0.5 m)")
    ap.add_argument("--home-at", default="0,0", metavar="X,Y", help="landing spot, start frame (default 0,0)")
    ap.add_argument("--mission", choices=["search"], default=None,
                    help="search = the real-area mission: outward ellipse, backup ellipses for missing targets, "
                         "land on the pad (config.yaml search_area:; no --trajectory needed)")
    ap.add_argument("--targets", type=int, default=None, help="search: targets to find besides home + anchor "
                    "(default search_area.targets = 3)")
    ap.add_argument("--blind-home", action="store_true",
                    help="experiment: fly straight back to (0, 0) by PX4 only - no tag-20 look, no drift "
                         "correction, no pad re-aim (same as config tuning blind_home: true)")
    ap.add_argument("--no-track-loop", action="store_true",
                    help="search: do not use the correction loop even if the drone's firmware has it")
    ap.add_argument("--search-retries", type=int, default=1,
                    help="search: backup ellipses at most when targets are missing (default 1)")
    ap.add_argument("--pad-tries", type=int, default=2,
                    help="search: after flying onto the pad, correction moves at most if it settled off-centre "
                         "(default 2; 0 = land straight away)")
    ap.add_argument("--pad-tolerance", type=float, default=0.12,
                    help="search: over the pad, this close to the centre is good enough (m, default 0.12)")
    ap.add_argument("--land-at-end", action="store_true",
                    help="land as soon as the path ends: no tag-20 look or correction moves (outward_rectangle "
                         "paths do this anyway)")
    ap.add_argument("--start-return", action="store_true",
                    help="after take-off, fly back onto (0, 0) before the trajectory (START_MOVE ...; 10-30 s).  "
                         "Off by default since 2 Nov: the path is shifted back onto the start mark instead")
    ap.add_argument("--start-tolerance", type=float, default=0.05,
                    help="after take-off the drone flies back onto (0, 0) and the trajectory starts only when "
                         "it is within this many metres (default 0.05)")
    ap.add_argument("--start-tries", type=int, default=4, help="moves back to (0, 0) at most (default 4)")
    ap.add_argument("--start-settle-s", type=float, default=1.5,
                    help="hover this long after each move before checking the position (default 1.5)")
    ap.add_argument("--start-samples", type=int, default=3,
                    help="PX4 samples (~1 s apart) averaged to judge the position after each move (default 3)")
    ap.add_argument("--aim-gain", type=float, default=0.8,
                    help="each move aims past the target by this share of what the last move missed (default 0.8; "
                         "0 = always aim at the target itself)")
    ap.add_argument("--aim-max", type=float, default=0.35, help="never aim more than this far past a target (m)")
    ap.add_argument("--start-max-move", type=float, default=1.0,
                    help="more than this far from (0, 0) after take-off = something is wrong: land (default 1.0)")
    ap.add_argument("--home-tolerance", type=float, default=0.04, help="good enough (m), default 0.04")
    ap.add_argument("--home-tries", type=int, default=3, help="correction moves at most (default 3)")
    ap.add_argument("--home-max-move", type=float, default=1.2, help="refuse bigger corrections (m)")
    ap.add_argument("--home-look-s", type=float, default=4.0, help="seconds to look for the tag per pass")
    ap.add_argument("--label", default="",
                    help="name at the front of the log folder, e.g. F1 -> logs/F1_comb_lawnmower_<date>_<time>")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--telem-port", type=int, default=5005)
    ap.add_argument("--cmd-port", type=int, default=5006)
    ap.add_argument("--camera-port", type=int, default=5009)
    return ap


def _set_in_yaml_text(text: str, dotted: str, value) -> tuple[str, bool]:
    """Change one value in YAML text, keeping every comment (for config_used.yaml)."""
    import re
    import yaml
    lines = text.splitlines(keepends=True)
    start, end, indent = 0, len(lines), -1
    keys = dotted.split(".")
    for depth, k in enumerate(keys):
        hit = None
        for i in range(start, end):
            m = re.match(r"^(\s*)" + re.escape(k) + r"\s*:", lines[i])
            if m and len(m.group(1)) > indent:
                hit, ind = i, len(m.group(1))
                break
        if hit is None:
            return text, False
        block_end = end
        for j in range(hit + 1, end):
            t = lines[j]
            if t.strip() and not t.lstrip().startswith("#") and len(t) - len(t.lstrip()) <= ind:
                block_end = j
                break
        if depth == len(keys) - 1:
            val = yaml.safe_dump(value, default_flow_style=True).strip()
            if val.endswith("..."):
                val = val[:-3].strip()
            lines[hit] = " " * ind + f"{k}: {val}   # set for this run\n"
            return "".join(lines), True
        start, end, indent = hit + 1, block_end, ind
    return text, False


def write_config_used(src, dst: Path, mount, overrides=()) -> None:
    """Copy the config with the values this run actually used (mount + --set changes)."""
    text = Path(src).read_text()
    changed = list(overrides) + [("camera.mount.pitch_deg", round(mount.pitch_deg, 6)),
                                 ("camera.mount.offset_fwd_m", round(mount.offset_fwd_m, 6)),
                                 ("camera.mount.offset_right_m", round(mount.offset_right_m, 6)),
                                 ("camera.mount.offset_down_m", round(mount.offset_down_m, 6))]
    for key, val in changed:
        text, _ = _set_in_yaml_text(text, key, val)
    dst.write_text(text)


def default_mount_config() -> Path:
    """The drone's camera mount: set_camera_mount.py's file when it exists, else the 29 Oct one."""
    here = HERE / "optimization_comparison" / "laptop"
    cur = here / "mapper_config_current.yaml"
    return cur if cur.exists() else here / "mapper_config_29_Oct.yaml"


def write_mapper_config_used(dst: Path, mount, base: Path | None) -> Path:
    """The optimisation's mapper config for this run, with the camera mount this run used."""
    import yaml
    here = HERE / "optimization_comparison" / "laptop"
    if base is None or not Path(base).exists():
        base = next((p for p in (here / "mapper_config_current.yaml", here / "mapper_config_29_Oct.yaml",
                                 here / "mapper_config.example.yaml") if p.exists()), None)
    if base is None:
        return None
    c = yaml.safe_load(Path(base).read_text())
    p = math.radians(mount.pitch_deg)
    sp, cp = math.sin(p), math.cos(p)
    ext = c["camera"]["extrinsic_T_B_C"]
    old = (list(map(float, ext.get("translation_m") or [])), ext.get("rotation_matrix"))
    ext["translation_m"] = [float(mount.offset_fwd_m), float(mount.offset_right_m), float(mount.offset_down_m)]
    ext["rotation_matrix"] = [[0.0, -sp, cp], [1.0, 0.0, 0.0], [0.0, cp, sp]]
    if ext.get("mount_id") == "provisional_sketch_mount" or old[0] != ext["translation_m"] or not all(
            abs(a - b) < 1e-6 for ra, rb in zip(old[1] or [], ext["rotation_matrix"]) for a, b in zip(ra, rb)):
        ext["mount_id"] = "run_mount"
        ext["measurement_notes"] = (f"mount used by this run: {mount.offset_fwd_m} m fwd, {mount.offset_right_m} m "
                                    f"right, {mount.offset_down_m} m down, pitch {mount.pitch_deg} deg "
                                    f"(base {Path(base).name})")
    dst.write_text(yaml.safe_dump(c, sort_keys=False))
    return dst


def apply_camera_settings(args, changes, rec, out_dir: Path) -> None:
    """Send the configured camera chip settings (if any) and record what the camera is using.
    Never stops the run: firmware without the camera-settings command just does not answer."""
    import json
    import camera_settings as camset
    try:
        r = camset.request(args.esp_ip, changes, port=args.cmd_port,
                           timeout=0.8 if changes else 0.5, tries=3 if changes else 1)
    except camset.CameraSettingsError:
        if changes:
            rec.note("camera settings NOT applied (" + " ".join(f"{n}={v}" for n, v in changes) +
                     "): no reply - this drone's firmware does not have the camera-settings command yet")
        return
    bad = [f"{n} {camset.STATUS.get(st, st)}" for n, st in r["results"] if st != 0]
    if changes:
        rec.note("camera settings applied: " + " ".join(f"{n}={v}" for n, v in changes) +
                 (f"  (problems: {', '.join(bad)})" if bad else ""))
    p = r["params"]
    rec.note(f"camera: exposure {'auto' if p.get('aec') else 'manual %s' % p.get('aec_value')}, "
             f"ae_level {p.get('ae_level')}, brightness {p.get('brightness')}, contrast {p.get('contrast')}"
             + (f", live exposure {r['live_exposure']}" if r["live_exposure"] is not None else "")
             + (f", live gain x{r['live_gain']:.1f}" if r["live_gain"] is not None else ""))
    try:
        (out_dir / "camera_settings.json").write_text(json.dumps(
            dict(requested=dict(changes), reply=r), indent=2))
    except OSError:
        pass


def run_folder_name(args, stamp: str) -> str:
    """logs/<label or mode>_<trajectory name>[_h0.8][_yaw]_<date>_<time>, e.g.
    F1_comb_lawnmower_20261029_173012, so each run is easy to find."""
    import re
    parts = [args.label or args.mode]
    if args.mode == "fly" and args.trajectory:
        parts.append(Path(args.trajectory).stem)
        if args.height and abs(args.height - 0.5) > 1e-6:
            parts.append(f"h{args.height:g}")
        if getattr(args, "yaw_follow", False):
            parts.append("yaw")
        if getattr(args, "pitch", None) is not None:
            parts.append(f"p{args.pitch:g}")
    parts.append(stamp)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(str(x) for x in parts))


def resolve_anchors(args) -> None:
    """--home-tag/--home-tag-at/--anchor/--anchor-at not given -> the config.yaml anchors section."""
    import yaml
    try:
        cfg = (yaml.safe_load(Path(args.config).read_text()) or {}).get("anchors") or {}
    except OSError:
        cfg = {}

    def xy(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.split(",")
        x, y = (float(q) for q in v)
        return f"{x:g},{y:g}"
    ht = args.home_tag if args.home_tag is not None else cfg.get("home_tag")
    args.home_tag = None if ht is None or str(ht).strip().lower() in ("none", "off", "") else int(ht)
    args.home_tag_at = xy(args.home_tag_at) or xy(cfg.get("home_tag_at")) or "0.75,0"
    an = args.anchor if args.anchor is not None else cfg.get("map_tag")
    args.anchor = "auto" if an is None else str(an).strip().lower()
    if args.anchor_at is None and args.anchor == str(cfg.get("map_tag")):
        args.anchor_at = xy(cfg.get("map_tag_at"))
    else:
        args.anchor_at = xy(args.anchor_at)
    if args.home_tag is not None and args.anchor == str(args.home_tag):
        raise SystemExit(f"the map anchor and the home tag are both {args.home_tag}; use two different tags")
    # 5 Nov: more map anchors (config anchors: extra_map_tags), e.g. {11: (2, -1), 12: (2, 1)}
    args.extra_anchors = {}
    for k, v in (cfg.get("extra_map_tags") or {}).items():
        if str(k) not in (str(args.anchor), str(args.home_tag)):
            args.extra_anchors[int(k)] = tuple(float(q) for q in v)
    try:                                              # the project's tag16h5 table is cut down to IDs 0-29 (5 Nov)
        from apriltag_host import EXPECTED_NCODES as ncodes
    except Exception:                                 # noqa: BLE001
        ncodes = 30
    for what, tid in (("home tag", args.home_tag), ("map anchor", args.anchor)):
        if tid is not None and str(tid).isdigit() and int(tid) >= ncodes:
            print(f"\n*** WARNING: {what} {tid} can NOT be detected: this project's AprilTag detector only knows "
                  f"tag16h5 IDs 0-{ncodes - 1}.  Print a tag with an ID 0-{ncodes - 1} that is not used and set it in "
                  f"config.yaml (anchors:) or with --{'home-tag' if what == 'home tag' else 'anchor'}. ***\n")


def resolve_trajectory(args) -> None:
    """--trajectory CSV, or --trajectory SHAPE LENGTH WIDTH -> made now by make_trajectory.py."""
    if not args.trajectory:
        args.trajectory = None
        return
    words = list(args.trajectory)
    if len(words) == 1:
        args.trajectory = words[0]
        return
    import make_trajectory as mt
    if words[0] not in mt.SHAPES or len(words) != 3:
        raise SystemExit(f"--trajectory: give a CSV file, or SHAPE LENGTH WIDTH with SHAPE one of "
                         f"{', '.join(mt.SHAPES)} (e.g. --trajectory ellipse 5 2); got {' '.join(words)}")
    try:
        length, width = float(words[1]), float(words[2])
    except ValueError:
        raise SystemExit(f"--trajectory {' '.join(words)}: length and width must be numbers in metres")
    try:
        speed = float(args.speed) if args.speed not in ("", None) else 0.2
    except ValueError:
        speed = 0.2
    try:
        path = mt.make(words[0], length, width, out_dir=HERE / "trajectories", speed=speed,
                       height=max(0.5, float(args.height or 0.5)))
    except ValueError as e:
        raise SystemExit(f"--trajectory {' '.join(words)}: {e}")
    print(f"trajectory {words[0]} {length:g} m x {width:g} m made by make_trajectory.py -> {path}")
    args.trajectory = str(path)


def main() -> int:
    args = build_parser().parse_args()
    resolve_trajectory(args)
    resolve_anchors(args)
    area = None
    if args.mission == "search":
        import search_mission as _sm
        area = _sm.load_area(args.config, args.targets)
        args.anchor_at = f"{area.anchor_at[0]:g},{area.anchor_at[1]:g}"   # the anchor's spot in THIS area
        args.extra_anchors = {k: v for k, v in area.anchors.items() if str(k) != str(args.anchor)}
        if args.trajectory is None:
            args.trajectory = "search_mission"
    if args.mode == "fly" and not args.trajectory:
        raise SystemExit("--mode fly needs --trajectory")
    from miniproject.settings import parse_overrides
    try:
        overrides = parse_overrides(args.set)
        S = load_settings(args.config, overrides)
    except ValueError as e:
        raise SystemExit(str(e))
    import camera_settings as camset
    try:
        merged = dict(camset.parse_assignments(f"{k}={v}" for k, v in S.camera_sensor.items()))
        merged.update(camset.parse_assignments(args.sensor))
    except camset.CameraSettingsError as e:
        raise SystemExit(f"camera setting: {e}")
    sensor_changes = list(merged.items())
    mount_note = f"camera mount from {Path(args.config).name}"
    if args.mount_config and args.mount_config != "config":
        if Path(args.mount_config).exists():
            import dataclasses
            from miniproject.settings import mount_from_mapper
            S = dataclasses.replace(S, mount=mount_from_mapper(args.mount_config))
            mount_note = f"camera mount from {Path(args.mount_config).name}"
        else:
            mount_note += f" ({args.mount_config} not found)"
    # camera.mount values given with --set, then the quick flags, win over the mount file
    import dataclasses
    quick = {k.split(".")[-1]: v for k, v in overrides if k.startswith("camera.mount.")}
    for flag, field in (("pitch", "pitch_deg"), ("cam_fwd", "offset_fwd_m"), ("cam_down", "offset_down_m")):
        if getattr(args, flag) is not None:
            quick[field] = getattr(args, flag)
    if quick:
        S = dataclasses.replace(S, mount=dataclasses.replace(S.mount, **{k: float(v) for k, v in quick.items()}))
        mount_note += " + command line (" + ", ".join(f"{k}={v:g}" for k, v in quick.items()) + ")"
    if not 0.0 <= S.mount.pitch_deg <= 90.0:
        raise SystemExit(f"camera pitch {S.mount.pitch_deg} deg is outside 0-90 (nose-down positive)")
    mount_note += (f": {S.mount.offset_fwd_m:.3f} m forward, {S.mount.offset_right_m:.3f} m right, "
                   f"{S.mount.offset_down_m:.3f} m down, pitch {S.mount.pitch_deg:.1f} deg nose-down")
    if not 0.5 <= args.height <= 1.1:
        raise SystemExit("--height must be between 0.5 and 1.1 m (firmware takes off to 0.5 m; it allows +0.6 m)")
    if args.height > 0.5:                     # camera is higher -> tags are further away
        import dataclasses
        extra = args.height - 0.5
        S = dataclasses.replace(S, gate=dataclasses.replace(S.gate, max_range_m=S.gate.max_range_m + extra / 3.0))
    refs = load_refs(args.refs)                      # the TARGET tags (tape truth), e.g. 0-3
    expected = parse_ids(args.expect_ids) or sorted(refs)
    known = dict(refs)                                # targets + the map anchor: for the ref errors
    if args.anchor not in ("auto", "home"):
        aid = int(args.anchor)
        if args.anchor_at:
            known[aid] = tuple(float(v) for v in args.anchor_at.split(","))
        elif aid not in refs:
            print(f"note: no position for anchor tag {aid} (--anchor-at or anchors.map_tag_at in config.yaml): "
                  f"the optimisation will anchor on the farthest target tag instead")
    known.update(getattr(args, "extra_anchors", {}) or {})

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_folder_name(args, stamp)
    rec = Recorder(out_dir)
    write_config_used(args.config, out_dir / "config_used.yaml", S.mount, overrides)
    args.mapper_config_used = write_mapper_config_used(
        out_dir / "mapper_config_used.yaml", S.mount,
        Path(args.mount_config) if args.mount_config and args.mount_config != "config" else None)
    rec.note(f"logging to {out_dir}")
    rec.note(mount_note)
    for k, v in overrides:
        if not k.startswith("camera.mount."):
            rec.note(f"for this run: {k} = {v}")
    g = S.gate
    rec.note(f"settings: range limit {g.max_range_m:.2f} m, margin > {g.min_margin:g}, hamming <= {g.max_hamming}, "
             f"confirm after {S.min_obs} frames, height {args.height:g} m")
    rec.note(f"tags: home tag {args.home_tag if args.home_tag is not None else '-'}"
             + (f" at ({args.home_tag_at})" if args.home_tag is not None else "")
             + f", map anchor {args.anchor}" + (f" at ({args.anchor_at})" if args.anchor_at else "")
             + f", targets {sorted(refs) or '-'}")
    mav = None
    if args.mavlink_port:
        from miniproject.mavlink_source import MavlinkSource
        mav = MavlinkSource(out_dir, args.mavlink_port, note=rec.note).start()
        rec.mav = mav
        if mav.ok:
            rec.note(f"MAVLink: listening on UDP {args.mavlink_port} -> {out_dir / 'mavlink.csv'}")

    history = PoseHistory()
    tag_map = TagMap(S.inlier_radius_m, min_frames=S.min_obs)
    link = Link(args.drone_id, history, S.timing.pose_latency_s, rec)

    def first_seen(o) -> None:                       # candidate only: NOT a confirmed target
        rec.note(f"candidate id {o.tag_id}: accepted frame 1/{S.min_obs}  (frame {o.frame_id}, "
                 f"range {o.range_m:.2f} m)")

    def confirmed(o) -> None:                        # printed exactly once per ID per mission
        e = tag_map.estimate(o.tag_id)
        txt = (f"id {o.tag_id}  N {e.north:+.3f}  E {e.east:+.3f}  after {o.accepted_frame_count} accepted "
               f"frames (first seen T+{(rec.since_t0(e.first_t_epoch) or 0):.2f} s)")
        if o.tag_id in known:
            rn, re_ = known[o.tag_id]
            txt += f"  (ref error {math.hypot(e.north - rn, e.east - re_):.3f} m)"
        rec.event("TAG_CONFIRMED", txt, tag_id=o.tag_id, north=e.north, east=e.east,
                  accepted_frame_count=o.accepted_frame_count, confirmed_at=o.confirmed_at)

    pipeline = TagPipeline(S, history, tag_map, rec, on_first_detection=first_seen, refs=known,
                           on_confirmed=confirmed)
    state_home = []                                   # (t_mono, obs) of accepted HOME tag sightings
    state_targets = []                                # (t_epoch, id, north, east) of accepted target sightings
    if args.home_tag is not None:
        if args.home_tag in known:
            raise SystemExit(f"home tag {args.home_tag} is also a target or the map anchor; use another ID")
        pipeline.special_ids.add(args.home_tag)

        def home_listener(o) -> None:
            if o.tag_id == args.home_tag and o.accepted and o.fwd is not None:
                state_home.append((time.monotonic(), o))
                del state_home[:-200]
        pipeline.listeners.append(home_listener)

        def target_listener(o) -> None:
            if o.accepted and o.tag_id != args.home_tag and o.north is not None:
                state_targets.append((o.t_epoch, o.tag_id, o.north, o.east))
        pipeline.listeners.append(target_listener)
    extra_ids = [i for i in ([args.home_tag] + [k for k in known if k not in refs]) if i is not None]
    if S.gate.allowed_ids and any(i not in S.gate.allowed_ids for i in extra_ids):
        import dataclasses
        S = dataclasses.replace(S, gate=dataclasses.replace(
            S.gate, allowed_ids=tuple(S.gate.allowed_ids) + tuple(i for i in extra_ids if i not in S.gate.allowed_ids)))
        pipeline.s = S
    view = None
    if args.show or args.record_video:
        from miniproject.video_view import VideoView

        def view_header() -> str:
            p = link.pkt
            if p is None:
                return ""
            n, e = pipeline._to_origin(p.ned_x, p.ned_y)
            return f"pos from start N {n:+.2f} E {e:+.2f}"
        view = VideoView(out_dir, show=args.show, record=args.record_video, fps=10, min_frames=S.min_obs,
                         rec=rec, header_fn=view_header)
    cam = CameraWorker(args.esp_ip, S, pipeline, rec, camera_port=args.camera_port, view=view)

    comms = CommsNode(listen_port=args.telem_port, cmd_port=args.cmd_port, config_path=None)
    global OFFSET_CLEARED_AT
    comms.set_nav_tags([], {})
                                                # start offset (0,0): telemetry is the drone's LOCAL frame
    comms.on_telemetry(link.callback)

    def on_sigint(signum, frame):     # noqa: ARG001
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, on_sigint)

    state = {"start_sent": False, "land_sent": False, "rows": None, "home_sightings": state_home,
             "targets": state_targets, "area": area}
    code = 0
    comms.start()
    try:
        rec.note(f"waiting for telemetry from drone {args.drone_id} ...")
        link.wait_first(S.flight.connect_timeout_s, comms.known_drones)
        rec.note("telemetry link established")
        apply_camera_settings(args, sensor_changes, rec, out_dir)
        comms.send_nav_tags(args.drone_id)       # push the (0, 0) start offset now, not on the lab's slow resend timer
        OFFSET_CLEARED_AT = time.monotonic()
        if view is not None:
            view.start()
            view.ready.wait(10)
            if args.show:
                rec.note("video window open: click back on THIS terminal before typing.  "
                         "In the video window: q = close it, l = LAND (same as Ctrl+C)")
        cam.start()
        cam.ready.wait(90)
        if cam.error:
            raise RuntimeError(f"camera pipeline failed to start: {cam.error}")
        rec.note("camera pipeline ready (detector built)")
        if args.mode == "observe":
            run_observe(args, S, link, cam, rec, tag_map)
        else:
            if args.mission == "search":
                run_search(args, S, comms, link, cam, rec, tag_map, known, state)
            else:
                run_fly(args, S, comms, link, cam, rec, tag_map, known, state)
    except KeyboardInterrupt:
        rec.event("INTERRUPTED", "Ctrl+C")
        code = 130
    except Exception as exc:          # noqa: BLE001 - report everything, then land if we flew
        if type(exc).__name__ == "GuardTrip":
            rec.event("GUARD_LAND", str(exc))
            if state.get("guard") is not None:
                rec.note(state["guard"].summary())
        else:
            rec.event("ABORTED", f"{type(exc).__name__}: {exc}")
        code = 1
    finally:
        if state["start_sent"] and not state["land_sent"]:
            send_land(comms, args.drone_id, rec, "abort/fallback")
        if state.get("track_loop") is not None:
            state["track_loop"].stop()
            rec.note(state["track_loop"].summary())
        if mav is not None:
            rec.note(mav.summary())
            mav.stop()
        cam.stop()
        if cam.is_alive():
            cam.join(timeout=3)
        if view is not None:
            view.stop()
            view.join(timeout=15)
            rec.note(view.summary())
        comms.stop()
        rows = tag_table(rec, tag_map, known, "FINAL", S.min_obs)
        home_rows = None
        d0 = home_drift_start_or_zero(state, rec, args.home_tag)
        if d0 and state.get("home_drift_end"):
            confirmed_ids = set(tag_map.confirmed_ids())
            obs_ok = [x for x in state["targets"] if x[1] in confirmed_ids]
            home_rows = drift_corrected_table(rec, known, obs_ok, d0, state["home_drift_end"], args.home_tag)
            write_home_corrected_csv(out_dir / "home_corrected_tags.csv", home_rows)
        elif args.home_tag is not None and args.mode == "fly":
            rec.note(f"no tag-{args.home_tag} drift correction: the home tag was not seen back home")
        metrics = None
        if args.mode == "fly":
            try:
                metrics = flight_metrics(rec, state, rows, refs, d0 if d0 else None,
                                         state.get("home_drift_end"), out_dir,
                                         yaw_follow=bool(getattr(args, "yaw_follow", False)))
            except Exception as exc:              # noqa: BLE001 - never lose the log over a metric
                rec.note(f"flight metrics failed: {type(exc).__name__}: {exc}")
        if state.get("track"):
            with open(out_dir / "track.csv", "w", encoding="utf-8") as tf:
                tf.write("t_s,north_start_m,east_start_m,px4_x_m,px4_y_m\n")
                for r in state["track"]:
                    tf.write(",".join(f"{v:.3f}" for v in r) + "\n")
        if state.get("t_takeoff") and state.get("t_land"):
            rec.event("MISSION_SUMMARY", f"mission time {state['t_land'] - state['t_takeoff']:.1f} s "
                      f"(takeoff command -> land command), {sum(1 for r in rows.values() if r.get('confirmed'))} confirmed tag(s)")
        t_ref = rec.t0 or time.time()
        ws = write_worksheet_rows(rec, rows, expected, args, t_ref)
        rec.write_json("summary.json", {
            "args": {k: str(v) for k, v in vars(args).items()},
            "tags": rows, "frames_processed": pipeline.frames,
            "rejected_by_reason": pipeline.rejected_by_reason,
            "map_anchor": {"id": args.anchor, "at": args.anchor_at},
            "home_tag": None if args.home_tag is None else {
                "id": args.home_tag, "at": args.home_tag_at,
                "drift_start": d0, "drift_end": state.get("home_drift_end"),
                "rezeroed_after_takeoff": bool(state.get("home_drift_start")),
                "heading_error_deg": getattr(state.get("home"), "heading_error_deg", None),
                "drift_corrected_tags": home_rows},
            "start_return": state.get("start_return"),
            "search_found": state.get("search_found"),
            "search_home": state.get("search_home"),
            "metrics": metrics,
            "camera_fps_last": round(cam.fps, 2)})
        rec.note(f"frames processed {pipeline.frames}; rejected: {pipeline.rejected_by_reason or 'none'}")
        rec.note(f"files in {out_dir}: events.jsonl observations.csv summary.json {ws.name}"
                 + (" home_corrected_tags.csv" if home_rows else ""))
        rec.close()
    if args.mode == "fly":
        try:                                   # the real lens tilt, measured from this flight's tags
            from check_camera_tilt import report as tilt_report
            print("")
            tilt_report(out_dir)
        except Exception as exc:               # noqa: BLE001 - a check, never stop the run
            print(f"camera tilt check skipped: {type(exc).__name__}: {exc}")
    opt_rc = None
    if args.mode == "fly" and not args.no_optimize:
        opt_rc = run_optimisation(out_dir, args)
    if args.mission == "search":
        print_search_summary(out_dir, args, area, opt_rc=opt_rc)
    return code


def run_optimisation(out_dir: Path, args):
    """After landing: pose-graph optimisation of this flight (optimization_comparison bundle).

    Runs the team's before/after calculation on this run's own log folder, prints its terminal
    table, and writes <run>/optimization/ (comparison.csv, results.csv, before_after.png, ...).
    Never affects the flight: it starts only after the land command and the final tag report."""
    import os
    import subprocess
    import tempfile
    bundle = HERE / "optimization_comparison"
    script = bundle / "laptop" / "analyze_observations.py"
    print("\n" + "=" * 78 + "\nPOSE-GRAPH OPTIMISATION (after landing)\n" + "=" * 78, flush=True)
    if not script.exists():
        print(f"skipped: {script} not found (unzip the optimisation bundle into laptop\\optimization_comparison)")
        return
    if not args.refs:                 # search runs: no target truth, only the known tags
        empty = out_dir / "refs_none.csv"
        empty.write_text("# no target truth for this run\n")
        args.refs = str(empty)
    raw = out_dir / "raw_detections.csv"
    n_acc = 0
    if raw.exists():
        import csv as _csv
        with open(raw, newline="") as fh:
            n_acc = sum(1 for r in _csv.DictReader(fh)
                        if str(r.get("accepted", "")).strip().lower() in ("1", "true", "yes"))
    if n_acc == 0:
        print("skipped: no accepted tag detections in this flight, nothing to optimise")
        return
    out = out_dir / "optimization"
    if out.exists():
        print(f"skipped: {out} already exists")
        return
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    extra = []
    if getattr(args, "mapper_config_used", None):
        extra = ["--mapper-config", str(args.mapper_config_used)]
    refs_path = Path(args.refs).resolve()
    anchor = str(args.anchor)
    # the known tags' tape positions join the truth file: the map anchor (pins the map) and the
    # home tag (checked, never the anchor)
    add = {}
    try:
        from apriltag_host import EXPECTED_NCODES as ncodes
    except Exception:                                 # noqa: BLE001
        ncodes = 30
    if anchor not in ("auto", "home") and int(anchor) >= ncodes:
        print(f"map anchor {anchor} is outside the detector's IDs 0-{ncodes - 1} (never seen): anchoring on the "
              f"farthest target tag seen instead")
        anchor = "auto"
    elif args.anchor_at and anchor not in ("auto", "home"):
        add[int(anchor)] = tuple(float(v) for v in args.anchor_at.split(","))
    for _aid, _at in (getattr(args, "extra_anchors", {}) or {}).items():   # 5 Nov: tags 11, 12
        if _aid < ncodes:
            add[int(_aid)] = tuple(_at)
    if args.home_tag is not None:
        add[args.home_tag] = tuple(float(v) for v in args.home_tag_at.split(","))
    if add:
        lines = [l for l in refs_path.read_text().splitlines()
                 if not any(l.strip().startswith(f"{k},") for k in add)]
        lines += [f"{k},{x:g},{y:g}" for k, (x, y) in add.items()]
        refs_path = out_dir / "refs_with_anchors.csv"
        refs_path.write_text("\n".join(lines) + "\n")
    if args.home_tag is not None:
        hx, hy = add[args.home_tag]
        if anchor in ("auto", "home", str(args.home_tag)):
            # Tested 30 Oct: anchoring the solver on a tag only 0.45 m from the start leaves the
            # map free to swing about it (tags ended up 4-6 m off).  Anchor the confirmed truth
            # tag FARTHEST from the start instead; the home tag stays in as a checked tag.
            far = None
            try:
                import json as _json
                tags = _json.loads((out_dir / "summary.json").read_text()).get("tags", {})
                refs = load_refs(args.refs)
                cands = [(math.hypot(*refs[int(k)]), int(k)) for k, v in tags.items()
                         if v.get("confirmed") and int(k) in refs and int(k) != args.home_tag]
                far = max(cands)[1] if cands else None
            except (OSError, ValueError, KeyError):
                far = None
            if anchor == str(args.home_tag):
                print(f"anchor: the home tag is too close to the start to anchor the solver; using "
                      f"{'tag %d' % far if far is not None else 'the best-seen tag'} (farthest confirmed) instead")
            anchor = str(far) if far is not None else "auto"
    if anchor not in ("auto", "home"):
        ax_, ay_ = add.get(int(anchor), load_refs(args.refs).get(int(anchor), (float("nan"), float("nan"))))
        print(f"map anchor: tag {anchor} at ({ax_:+.2f}, {ay_:+.2f}) pins the map "
              f"(if it was not seen, the farthest target tag seen is used instead)")
        extras = [k for k in (getattr(args, "extra_anchors", {}) or {}) if k in add]
        for k in extras:
            print(f"map anchor: tag {k} at ({add[k][0]:+.2f}, {add[k][1]:+.2f}) pins the map too (if seen)")
        if args.home_tag is not None and int(anchor) != args.home_tag:
            # 3 Nov: every known tag is a constraint.  The home tag alone is too close to the start
            # to hold the map (30 Oct), but next to an anchor 1.5+ m away it also pins the map's
            # rotation.  Every listed anchor is used only if seen in >= 3 frames.
            extras.append(args.home_tag)
            print(f"home tag {args.home_tag} at ({hx:+.2f}, {hy:+.2f}) is an anchor too (fixed constraint)")
        anchor = ",".join([str(anchor)] + [str(k) for k in extras])
    elif args.home_tag is not None:
        print(f"home tag {args.home_tag} at ({hx:+.2f}, {hy:+.2f}) added to the truth file (checked)")
    cmd = [sys.executable, str(script), str(out_dir), "--output", str(out), *extra,
           "--refs", str(refs_path), "--anchor", anchor]
    print("running:", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="tag-opt-mpl-") as mpl:
        env["MPLCONFIGDIR"] = mpl
        try:
            rc = subprocess.run(cmd, env=env, timeout=args.optimize_timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"optimisation stopped after {args.optimize_timeout:.0f} s (raise --optimize-timeout)")
            return None
        except KeyboardInterrupt:
            print("optimisation interrupted")
            return None
    print(f"optimisation finished in {time.time() - t0:.0f} s (exit code {rc}); results in {out}")
    shown = [out / n for n in ("map.png", "before_after.png", "terminal_results.txt") if (out / n).exists()]
    if shown:
        print("  graphs and logs to show the optimisation:")
        for pth in shown + [out / n for n in ("tags.csv", "comparison.csv", "residuals.csv", "iterations.csv")
                            if (out / n).exists()]:
            print(f"    {pth}")
        if getattr(args, "show", False) and hasattr(os, "startfile"):     # Windows: open the two pictures
            for pth in shown[:2]:
                try:
                    os.startfile(str(pth))                                  # noqa: S606
                except OSError:
                    pass
    if rc == 2:
        print("  the solver stopped before fully settling (iteration limit or weak data); its positions are "
              "shown below only for tags seen in >= 30 frames (tuning opt_min_frames) and inside the area")
    elif rc != 0:
        print("  the optimisation failed (the flight log above is still complete); see the error "
              f"above and {out / 'failure.json'} if present")
    return rc


if __name__ == "__main__":
    sys.exit(main())
