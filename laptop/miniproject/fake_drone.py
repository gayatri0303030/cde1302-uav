#!/usr/bin/env python3
"""A fake ESP32 drone on localhost, for rehearsing the laptop software without hardware.

It speaks the real wire protocol (laptop/protocol.py):
  * telemetry  -> UDP :5005 at 10 Hz (position, heading, nav state),
  * commands   <- UDP :5006 (CMD_START, trajectory upload + start, CMD_LAND, camera keepalive,
                  CMD_SET_NAV_TAGS start offset),
  * camera     -> UDP :5009, JPEG frames in ECAM datagrams, rendered from a synthetic floor with
                  real tag16h5 markers, each frame showing the pose the drone had `frame_age_ms`
                  BEFORE it was sent (so the laptop's time sync is exercised for real).

It starts with a deliberately WRONG start offset (like a drone left over from a map-frame
run) so you can see the laptop clear it.

    python3 laptop/miniproject/fake_drone.py          # then, in another terminal:
    python3 laptop/run_baseline.py --drone-id 5 --esp-ip 127.0.0.1 --mode observe
    python3 laptop/run_baseline.py --drone-id 5 --esp-ip 127.0.0.1 --mode fly \
        --trajectory trajectory/ellipse_ours.csv

Not modelled: obstacles, PX4 dynamics (the pose follows the commanded path exactly),
roll/pitch, Wi-Fi loss.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

if __package__ in (None, ""):                          # run as a script: put laptop/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from miniproject.fake_drone import main            # noqa: E402
    raise SystemExit(main())

import cv2

from protocol import (
    CAMERA_HEADER_FMT, CAMERA_MAGIC, CAMERA_STREAM_PORT, CAMERA_VERSION, CAMERA_FLAG_END,
    CMD_CAMERA_STREAM, CMD_LAND, CMD_SET_NAV_TAGS, CMD_START, CMD_TRAJ_DATA, CMD_TRAJ_START,
    NAV_ARRIVED, NAV_IDLE, NAV_TRAJ, PKT_CMD, PKT_TELEM, _TELEM_HDR_FMT,
)

from .geometry import CameraMount
from .sim_world import render_frame

DEFAULT_TAGS = {3: (0.55, 0.25), 5: (-0.20, -0.30), 13: (0.10, 0.00)}   # id -> (north, east) in local frame


class FakeDrone:
    def __init__(self, drone_id: int = 5, tags: dict | None = None, host: str = "127.0.0.1",
                 telem_port: int = 5005, cmd_port: int = 5006, cam_port: int = CAMERA_STREAM_PORT,
                 mount: CameraMount | None = None, tag_size_m: float = 0.12,
                 frame_age_ms: int = 120, fps: float = 9.0, hover_alt: float = 0.5,
                 takeoff_s: float = 4.0, land_s: float = 2.0, start_offset=(-2.0, -4.0),
                 noise: float = 1.5, verbose: bool = True, start_pose=(0.0, 0.0, 0.0),
                 full_telemetry: bool = False, mavlink_port: int = 0, camera_settings: bool = True,
                 glitch=None, drift=None, yaw_drift=None, takeoff_drift=None, hold_bias=None,
                 nose_bias=None, track_loop: bool = True):
        self.id = drone_id
        # (ahead, right, tau_s): while flying a trajectory the drone sits this far from its setpoint
        # in ITS OWN frame (turns with the nose) and PX4 sees it but does not pull back - like drone 5
        # on 3 Nov (0.2-0.3 m towards its nose).  First-order lag tau_s.
        self.nose_bias = nose_bias
        self._nb = [0.0, 0.0]
        # track-loop firmware: CMD_TRAJ_NUDGE (0x0B) offset added to every trajectory setpoint
        self.track_loop = track_loop
        self._nudge_tgt, self._nudge, self._nudge_t = [0.0, 0.0], [0.0, 0.0], 0.0
        # (dx, dy): the drone really slides this far (local frame) while it takes off, and PX4 sees
        # it, like the real drone ending up at (+0.14, -0.10) after take-off on 2 Nov.
        self.takeoff_drift = takeoff_drift
        # (dx, dy, tau_s): the drone settles this far from where it is told to go (local frame), with a
        # first-order lag tau_s, like drone 5 on 2 Nov (always ~0.10 m ahead, 0.15 m left of the target).
        # Every new trajectory starts from the MEASURED position, so the offset builds up again.
        self.hold_bias = hold_bias
        self._lag = [0.0, 0.0]
        self._lag_at_hold = (0.0, 0.0)
        self._takeoff_xy0 = (0.0, 0.0)
        # (deg, seconds): PX4's HEADING estimate drifts by deg over that many seconds after take-off.
        # PX4 holds its estimate, so the real nose turns the other way, and moves commanded along
        # PX4's axes go along axes turned by that error in the real room.
        self.yaw_drift = yaw_drift
        self._yaw_set = 0.0
        self._est = None                  # PX4 position estimate (local), None = same as the truth
        self._traj_true0 = (0.0, 0.0)
        self._traj_dn0 = (0.0, 0.0)
        self._hold_true = None
        self._hold_dn = (0.0, 0.0)
        # (t_s, dx, dy): t_s after CMD_START the REPORTED position jumps by (dx, dy), like a PX4
        # estimator reset (30 Oct F3).  The simulated drone itself does not move.
        self.glitch = glitch
        self._t_start_cmd = None
        # (dx, dy, seconds): PX4's estimate drifts away from the truth by up to (dx, dy) over that
        # many seconds after take-off.  PX4 flies its ESTIMATE along the path, so the real drone
        # ends up off by -drift (what the home tag has to fix).
        self.drift = drift
        self._hold_est = None
        # camera-settings firmware patch (CMD 0x0A); False = behave like firmware without it
        self.camera_settings = camera_settings
        self.sensor = dict(brightness=1, contrast=0, sharpness=0, denoise=0, ae_level=0, aec=1,
                           aec_value=0, agc=1, agc_gain=0, gainceiling=248, aec2=0, lenc=1, bpc=1,
                           wpc=0, raw_gma=0, vflip=1, hmirror=0)
        self.full_telemetry = full_telemetry   # also send z/velocity + attitude extension (new firmware)
        self.mavlink_port = mavlink_port       # >0: also stream PX4 MAVLink (QGC bridge) to this UDP port
        self.tags = dict(DEFAULT_TAGS if tags is None else tags)
        self.host, self.telem_port, self.cmd_port, self.cam_port = host, telem_port, cmd_port, cam_port
        self.mount = mount or CameraMount()
        self.tag_size_m = tag_size_m
        self.frame_age_ms = frame_age_ms
        self.fps = fps
        self.hover_alt, self.takeoff_s, self.land_s = hover_alt, takeoff_s, land_s
        self.offset = list(start_offset)                # what the drone ADDS to local to get "map" coordinates
        self.noise = noise
        self.verbose = verbose
        # state
        self.x, self.y, self.yaw = start_pose
        self.alt = 0.0
        self.state = "idle"                             # idle | takeoff | hover | traj | landing
        self.nav_state = NAV_IDLE
        self._t_phase = 0.0
        self._alt0 = 0.0
        self.traj: list | None = None
        self.traj_id = 0
        self.traj_dt = 0.05
        self._traj_t0 = 0.0
        self._traj_org = (0.0, 0.0)
        self.log: list[str] = []
        self.pose_log: deque = deque(maxlen=400)        # (t, x, y, yaw, alt)
        self._cam_until = 0.0
        self._cam_dest: tuple | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _say(self, text: str) -> None:
        self.log.append(text)
        if self.verbose:
            print(f"  [fake drone] {text}", flush=True)

    def start(self) -> "FakeDrone":
        loops = [self._cmd_loop, self._sim_loop, self._telem_loop, self._cam_loop]
        if self.mavlink_port:
            loops.append(self._mavlink_loop)
        for fn in loops:
            th = threading.Thread(target=fn, daemon=True, name=fn.__name__)
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=2)

    # ------------------------------------------------------------------ camera settings (0x0A)
    _CAM_IDS = {1: "brightness", 2: "contrast", 3: "sharpness", 4: "denoise", 5: "ae_level", 6: "aec",
                7: "aec_value", 8: "agc", 9: "agc_gain", 10: "gainceiling", 11: "aec2", 12: "lenc",
                13: "bpc", 14: "wpc", 15: "raw_gma", 16: "vflip", 17: "hmirror"}
    _CAM_RANGE = {"brightness": (-3, 3), "contrast": (-3, 3), "sharpness": (-3, 3), "denoise": (0, 8),
                  "ae_level": (-5, 5), "agc_gain": (0, 64), "aec_value": (0, 984), "gainceiling": (0, 1023)}

    def _camera_settings(self, data: bytes) -> bytes:
        """Same wire format as main/camera_settings.c (fake OV3660)."""
        op, n = (data[2], data[3]) if len(data) >= 4 else (0, 0)
        results = []
        for i in range(n if op == 1 else 0):
            pid, val = struct.unpack_from("<Bh", data, 4 + 3 * i)
            name = self._CAM_IDS.get(pid)
            if name is None or name in ("vflip", "hmirror"):
                results.append((pid, -2))
                continue
            lo, hi = self._CAM_RANGE.get(name, (0, 1))
            if name in ("aec", "agc", "aec2", "lenc", "bpc", "wpc", "raw_gma"):
                val = 1 if val else 0
            if name in ("aec_value", "agc_gain"):
                val = min(max(val, lo), hi)
            if not lo <= val <= hi:
                results.append((pid, -1))
                continue
            self.sensor[name] = val
            results.append((pid, 0))
            self._say(f"camera setting {name} = {val}")
        live_exp = self.sensor["aec_value"] if not self.sensor["aec"] else int(680 * 2 ** (self.sensor["ae_level"] / 2))
        live_gain = (self.sensor["agc_gain"] if not self.sensor["agc"] else 4) * 16
        out = struct.pack("<4sBBHiiiBB", b"ECSS", 1, self.id, 0x3660, live_exp, live_gain, 984,
                          len(self._CAM_IDS), len(results))
        out += b"".join(struct.pack("<Bh", i, self.sensor[nm]) for i, nm in self._CAM_IDS.items())
        out += b"".join(struct.pack("<Bb", i, st) for i, st in results)
        return out

    def _yawd_now(self) -> float:
        d = self.yaw_drift
        if not d or self._t_start_cmd is None:
            return 0.0
        return math.radians(d[0]) * min(1.0, (time.monotonic() - self._t_start_cmd) / max(1e-3, d[1]))

    def _est_now(self):
        """PX4's position estimate (local, without the map offset)."""
        if self._est is not None and self.state in ("hover", "traj", "landing"):
            return self._est
        dn = self._drift_now()
        return (self.x + dn[0], self.y + dn[1])

    def _step_lag(self):
        if not self.hold_bias:
            return 0.0, 0.0
        bx, by, tau = self.hold_bias
        a = min(1.0, 0.02 / max(1e-3, tau))
        self._lag[0] += (bx - self._lag[0]) * a
        self._lag[1] += (by - self._lag[1]) * a
        return self._lag[0], self._lag[1]

    def _drift_now(self):
        d = self.drift
        if not d or self._t_start_cmd is None:
            return 0.0, 0.0
        k = min(1.0, (time.monotonic() - self._t_start_cmd) / max(1e-3, d[2]))
        return d[0] * k, d[1] * k

    def _glitch_now(self):
        g = self.glitch
        if g and self._t_start_cmd is not None and time.monotonic() - self._t_start_cmd >= g[0]:
            return g[1], g[2]
        return 0.0, 0.0

    def _apply_sensor(self, img):
        """Rough look of the settings: brightness steps and exposure scale the picture."""
        import numpy as np
        sn = self.sensor
        scale = (sn["aec_value"] / 680.0) if not sn["aec"] else 2 ** (sn["ae_level"] / 4)
        out = img.astype(np.float32) * max(0.1, scale) + 12.0 * sn["brightness"]
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ commands
    def _cmd_loop(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.cmd_port))
        s.settimeout(0.2)
        incoming: dict[int, list] = {}
        while not self._stop.is_set():
            try:
                data, addr = s.recvfrom(2048)
            except socket.timeout:
                continue
            if len(data) < 2 or data[0] != PKT_CMD:
                continue
            cmd = data[1]
            with self._lock:
                if cmd == CMD_START:
                    if self.state == "idle":
                        self.state, self._t_phase, self._alt0 = "takeoff", time.monotonic(), self.alt
                        self._t_start_cmd = time.monotonic()
                        self._takeoff_xy0 = (self.x, self.y)
                        self._yaw_set = self.yaw
                        self._say("CMD_START -> OFFBOARD, armed, taking off")
                    else:
                        self._say("CMD_START ignored - not waiting for it")
                elif cmd == CMD_LAND:
                    if self.state not in ("idle", "landing"):
                        self.state, self._t_phase, self._alt0 = "landing", time.monotonic(), self.alt
                        self.nav_state = NAV_IDLE
                        self._say("CMD_LAND -> landing")
                elif cmd == 0x0A and self.camera_settings:
                    s.sendto(self._camera_settings(data), addr)
                elif cmd == 0x0B and self.track_loop and len(data) >= 10:
                    n, e = struct.unpack_from("<ff", data, 2)
                    m = math.hypot(n, e)
                    if not math.isfinite(m):
                        n = e = 0.0
                    elif m > 0.40:
                        n, e = n * 0.40 / m, e * 0.40 / m
                    self._nudge_tgt, self._nudge_t = [n, e], time.monotonic()
                    s.sendto(b"ENDG" + struct.pack("<ffff", n, e, *self._nudge), addr)
                elif cmd == CMD_CAMERA_STREAM:
                    if len(data) >= 5 and data[2]:
                        self._cam_until = time.monotonic() + 2.5
                        self._cam_dest = (addr[0], self.cam_port)
                    else:
                        self._cam_until = 0.0
                elif cmd == CMD_SET_NAV_TAGS:
                    _, _, count, sx, sy = struct.unpack_from("<BBBff", data, 0)
                    if (sx, sy) != tuple(self.offset):
                        self._say(f"start offset set to ({sx:+.2f}, {sy:+.2f})")
                    self.offset = [sx, sy]
                elif cmd == CMD_TRAJ_DATA:
                    _, _, tid, total, off, n = struct.unpack_from("<BBBHHB", data, 0)
                    pts = incoming.setdefault(tid, [None] * total)
                    for i in range(n):
                        pts[off + i] = struct.unpack_from("<fff", data, 8 + 12 * i)
                    self.traj_id = tid
                elif cmd == CMD_TRAJ_START:
                    _, _, tid, dt_ms = struct.unpack_from("<BBBH", data, 0)
                    flags = data[5] if len(data) >= 6 else 0      # bit0: yaw follows path (like the patched firmware)
                    pts = incoming.get(tid)
                    if self.state == "hover" and pts and all(p is not None for p in pts):
                        self.yaw_follow = bool(flags & 1)
                        # outward-yaw firmware: bit1 + two uint16 = points where the nose is 90 deg right of travel
                        self._out_win = (struct.unpack_from("<HH", data, 6)
                                         if (flags & 2) and len(data) >= 10 else None)
                        # yaw v2: limits (deg/s, deg/s^2) after the window; none = the FIRST yaw patch
                        # (0.1 rad per 50 ms tick = ~115 deg/s, no ramp)
                        if len(data) >= 12:
                            self._yaw_lim = (math.radians(data[10] or 30), math.radians(data[11] or 30))
                        else:
                            self._yaw_lim = None
                        self._yaw_rate, self._yaw_want = 0.0, self.yaw
                        self.traj, self.traj_dt = pts, dt_ms / 1000.0
                        dn = self._drift_now()
                        est = self._est_now()
                        self._traj_t0, self._traj_org = time.monotonic(), est
                        self._traj_true0, self._traj_dn0 = (self.x, self.y), dn
                        self._lag = [0.0, 0.0]
                        self._nb = [0.0, 0.0]                 # setpoint starts where it is: builds up again
                        self.state, self.nav_state = "traj", NAV_TRAJ
                        incoming.pop(tid, None)
                        self._say(f"trajectory {tid} started ({len(pts)} points, dt {dt_ms} ms)")
                    else:
                        self._say("trajectory start ignored (not hovering or upload incomplete)")
        s.close()

    # ------------------------------------------------------------------ physics (50 Hz)
    def _sim_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                if self.state == "takeoff":
                    k = min((now - self._t_phase) / self.takeoff_s, 1.0)
                    self.alt = self._alt0 + k * (self.hover_alt - self._alt0)
                    if self.takeoff_drift:
                        self.x = self._takeoff_xy0[0] + k * self.takeoff_drift[0]
                        self.y = self._takeoff_xy0[1] + k * self.takeoff_drift[1]
                    if k >= 1.0:
                        self.state = "hover"
                        self._hold_est = self._est_now()
                        self._hold_true, self._hold_dn = (self.x, self.y), self._drift_now()
                        self._lag, self._lag_at_hold = [0.0, 0.0], (0.0, 0.0)
                        self._say(f"altitude reached ({self.alt:.2f} m)")
                elif self.state == "hover" and self._hold_true is not None:
                    dn = self._drift_now()                      # PX4 holds its estimate still
                    self.x = self._hold_true[0] - (dn[0] - self._hold_dn[0])
                    self.y = self._hold_true[1] - (dn[1] - self._hold_dn[1])
                    self._est = self._hold_est
                    if self.hold_bias:
                        lx, ly = self._step_lag()
                        ex, ey = lx - self._lag_at_hold[0], ly - self._lag_at_hold[1]
                        self.x, self.y = self.x + ex, self.y + ey
                        self._est = (self._hold_est[0] + ex, self._hold_est[1] + ey)
                elif self.state == "traj":
                    k = (now - self._traj_t0) / self.traj_dt
                    last = len(self.traj) - 1
                    if k >= last:
                        px, py, _ = self.traj[last]
                        self.state, self.nav_state = "hover", NAV_ARRIVED
                        self._say("trajectory finished")
                    else:
                        i = int(k)
                        a = k - i
                        p0, p1 = self.traj[i], self.traj[i + 1]
                        px, py = p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1])
                        win = getattr(self, "_out_win", None)
                        in_win = win is not None and win[0] <= i < win[1]
                        if getattr(self, "yaw_follow", False) or in_win:
                            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                            if dx * dx + dy * dy > 1e-8:
                                self._yaw_want = math.atan2(dy, dx) + (math.pi / 2 if in_win else 0.0)
                            d = (getattr(self, "_yaw_want", self.yaw) - self.yaw + math.pi) % (2 * math.pi) - math.pi
                            lim = getattr(self, "_yaw_lim", None)
                            if lim is None:                                # first yaw patch: 2 rad/s, no ramp
                                step = 2.0 * 0.02
                                self.yaw = (self.yaw + max(-step, min(step, d)) + math.pi) % (2 * math.pi) - math.pi
                            else:                                          # v2: rate ramped, capped, stops on target
                                rmax, amax, dt = lim[0], lim[1], 0.02
                                vd = min(rmax, math.sqrt(2 * amax * abs(d)))
                                vd = math.copysign(min(vd, abs(d) / dt), d)
                                w = self._yaw_rate + max(-amax * dt, min(amax * dt, vd - self._yaw_rate))
                                self._yaw_rate = w
                                self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
                    if self.track_loop:                       # firmware: offset moves <= 0.15 m/s
                        tn, te = self._nudge_tgt if now - self._nudge_t <= 1.5 else (0.0, 0.0)
                        dn_, de_ = tn - self._nudge[0], te - self._nudge[1]
                        d_, lim_ = math.hypot(dn_, de_), 0.15 * 0.02
                        if d_ > lim_:
                            dn_, de_ = dn_ * lim_ / d_, de_ * lim_ / d_
                        self._nudge = [self._nudge[0] + dn_, self._nudge[1] + de_]
                        px, py = px + self._nudge[0], py + self._nudge[1]
                    dn = self._drift_now()                      # PX4 tracks its ESTIMATE
                    yd = self._yawd_now()
                    c, s_ = math.cos(-yd), math.sin(-yd)          # PX4's axes are turned by -yd in the room
                    rx, ry = px * c - py * s_, px * s_ + py * c
                    self.x = self._traj_true0[0] + rx - (dn[0] - self._traj_dn0[0])
                    self.y = self._traj_true0[1] + ry - (dn[1] - self._traj_dn0[1])
                    self._est = (self._traj_org[0] + px, self._traj_org[1] + py)
                    if self.hold_bias:
                        lx, ly = self._step_lag()
                        self.x, self.y = self.x + lx, self.y + ly
                        self._est = (self._est[0] + lx, self._est[1] + ly)
                    if self.nose_bias:                        # sits ahead/right of its nose, PX4 sees it
                        fa, fr, tau = self.nose_bias
                        c_, s_2 = math.cos(self.yaw), math.sin(self.yaw)
                        want = (fa * c_ - fr * s_2, fa * s_2 + fr * c_)
                        k_ = min(1.0, 0.02 / max(1e-3, tau))
                        self._nb = [self._nb[0] + (want[0] - self._nb[0]) * k_,
                                    self._nb[1] + (want[1] - self._nb[1]) * k_]
                        self.x, self.y = self.x + self._nb[0], self.y + self._nb[1]
                        self._est = (self._est[0] + self._nb[0], self._est[1] + self._nb[1])
                    if self.state == "hover":
                        self._hold_est = self._est
                        self._hold_true, self._hold_dn = (self.x, self.y), dn
                        self._lag_at_hold = tuple(self._lag)
                elif self.state == "landing":
                    k = min((now - self._t_phase) / self.land_s, 1.0)
                    self.alt = self._alt0 * (1.0 - k)
                    if k >= 1.0:
                        self.state = "idle"
                        self._est = None
                        self._say(f"touchdown, disarmed at TRUE ({self.x:+.3f}, {self.y:+.3f}), "
                                  f"nose {math.degrees(self.yaw):+.1f} deg")
                if self.yaw_drift and self.state in ("hover", "traj", "landing") and not getattr(self, "yaw_follow", False):
                    self.yaw = self._yaw_set - self._yawd_now()
                self.pose_log.append((now, self.x, self.y, self.yaw, self.alt))
            time.sleep(0.02)

    # ------------------------------------------------------------------ PX4 MAVLink bridge (50 Hz)
    def _mavlink_loop(self) -> None:
        from pymavlink import mavutil
        m = mavutil.mavlink_connection(f"udpout:{self.host}:{self.mavlink_port}", source_system=1,
                                       source_component=1, dialect="common")
        t0 = time.monotonic()
        last_hb = 0.0
        prev = None
        while not self._stop.is_set():
            now = time.monotonic()
            boot = int((now - t0) * 1000)
            with self._lock:
                x, y, yaw, alt = self.x, self.y, self.yaw, self.alt
            vx = vy = vz = 0.0
            if prev is not None and now > prev[0]:
                dt = now - prev[0]
                vx, vy, vz = (x - prev[1]) / dt, (y - prev[2]) / dt, (prev[3] - alt) / dt
            prev = (now, x, y, alt)
            try:
                if now - last_hb > 1.0:
                    m.mav.heartbeat_send(2, 12, 0, 0, 0)
                    last_hb = now
                m.mav.local_position_ned_send(boot, x, y, -alt, vx, vy, vz)
                m.mav.attitude_quaternion_send(boot, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2),
                                               0.0, 0.0, 0.0)
            except OSError:
                pass
            time.sleep(0.02)

    # ------------------------------------------------------------------ telemetry (10 Hz)
    def _telem_loop(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while not self._stop.is_set():
            with self._lock:
                pkt = struct.pack(_TELEM_HDR_FMT, PKT_TELEM, self.id,
                                  self._est_now()[0] + self.offset[0] + self._glitch_now()[0],
                                  self._est_now()[1] + self.offset[1] + self._glitch_now()[1],
                                  self.yaw + self._yawd_now(),
                                  self.nav_state, -1, 0.0, bytes(32), 0, 0xFFFF)
                if self.full_telemetry:
                    boot = int((time.monotonic() % 4e6) * 1000)
                    pkt += struct.pack("<HHHBBB", 20, 20, 20, 1, int(self.state != "idle"), 6)
                    pkt += struct.pack("<ffff", -self.alt, 0.0, 0.0, 0.0)
                    ye = self.yaw + self._yawd_now()             # PX4's (estimated) heading
                    pkt += struct.pack("<9fIIB", math.cos(ye / 2), 0.0, 0.0, math.sin(ye / 2),
                                       0.0, 0.0, 0.0, 0.0, 0.0, boot, boot, 1)
            try:
                s.sendto(pkt, (self.host, self.telem_port))
            except OSError:
                pass
            time.sleep(0.1)
        s.close()

    # ------------------------------------------------------------------ camera
    def _pose_at(self, t: float):
        with self._lock:
            log = list(self.pose_log)
        best = log[0] if log else (t, 0.0, 0.0, 0.0, 0.0)
        for rec in log:
            if rec[0] <= t:
                best = rec
            else:
                break
        return best

    def _cam_loop(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        frame_id = 0
        period = 1.0 / self.fps
        nonce = 77
        while not self._stop.is_set():
            t_send = time.monotonic()
            dest = self._cam_dest
            if dest is not None and t_send < self._cam_until:
                _, x, y, yaw, alt = self._pose_at(t_send - self.frame_age_ms / 1000.0)
                # a real hover is never perfectly still: +-1 cm / +-0.3 deg wobble (a perfectly
                # still, axis-aligned synthetic tag can alias and stop decoding)
                x += 0.01 * math.sin(1.7 * t_send)
                y += 0.01 * math.sin(2.3 * t_send + 1.0)
                yaw += 0.005 * math.sin(1.1 * t_send + 2.0)
                if alt > 0.05:                                   # nothing to see on the ground
                    img = render_frame(self.tags, (x, y, yaw), alt, self.mount, self.tag_size_m,
                                       noise=self.noise, seed=frame_id)
                else:
                    img = render_frame({}, (x, y, yaw), 0.3, self.mount, self.tag_size_m)
                img = self._apply_sensor(img)
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    data = jpg.tobytes()
                    frame_id += 1
                    chunk = 1000
                    for off in range(0, len(data), chunk):
                        part = data[off:off + chunk]
                        hdr = struct.pack(CAMERA_HEADER_FMT, CAMERA_MAGIC, CAMERA_VERSION, 0, self.id,
                                          nonce, frame_id, off, 0, 320, 240, len(part),
                                          self.frame_age_ms, 0)
                        s.sendto(hdr + part, dest)
                    end = struct.pack(CAMERA_HEADER_FMT, CAMERA_MAGIC, CAMERA_VERSION, CAMERA_FLAG_END,
                                      self.id, nonce, frame_id, len(data), len(data), 320, 240, 0,
                                      self.frame_age_ms, 0)
                    s.sendto(end, dest)
            time.sleep(max(0.0, period - (time.monotonic() - t_send)))
        s.close()


def parse_tags(text: str) -> dict:
    """'3:0.55,0.25;5:-0.2,-0.3' -> {3: (0.55, 0.25), 5: (-0.2, -0.3)}"""
    out = {}
    for item in filter(None, text.split(";")):
        tid, pos = item.split(":")
        n, e = pos.split(",")
        out[int(tid)] = (float(n), float(e))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fake ESP32 drone on localhost (no hardware needed)")
    ap.add_argument("--drone-id", type=int, default=5)
    ap.add_argument("--tags", default=None, help="'id:north,east;id:north,east' local-frame tag positions")
    ap.add_argument("--camera-pitch-deg", type=float, default=45.0)
    ap.add_argument("--frame-age-ms", type=int, default=120)
    ap.add_argument("--mavlink-port", type=int, default=0,
                    help="also stream PX4 MAVLink to this UDP port (14550 = like the QGC bridge)")
    ap.add_argument("--full-telemetry", action="store_true",
                    help="send z/velocity/quaternion/body rates like the patched firmware")
    ap.add_argument("--glitch", default=None, metavar="T,DX,DY",
                    help="T s after take-off the reported position jumps by DX,DY (estimator reset test)")
    ap.add_argument("--drift", default=None, metavar="DX,DY,SECONDS",
                    help="PX4 estimate drifts by DX,DY over SECONDS after take-off (home-tag test)")
    ap.add_argument("--yaw-drift", default=None, metavar="DEG,SECONDS",
                    help="PX4 heading estimate drifts by DEG over SECONDS after take-off")
    ap.add_argument("--no-camera-settings", action="store_true",
                    help="behave like firmware without the camera-settings command (no reply to 0x0A)")
    ap.add_argument("--takeoff-drift", default=None, metavar="DX,DY",
                    help="the drone slides DX,DY (local frame, m) during take-off and PX4 sees it")
    ap.add_argument("--hold-bias", default=None, metavar="DX,DY,TAU",
                    help="the drone settles DX,DY (local, m) off every target with lag TAU s (drone 5: 0.10,-0.15,1.5)")
    ap.add_argument("--nose-bias", default=None, metavar="AHEAD,RIGHT,TAU",
                    help="while flying a path the drone sits AHEAD,RIGHT m from its setpoint in its own frame "
                         "(turns with the nose), PX4 sees it and does not pull back (drone 5, 3 Nov: 0.25,-0.1,1.5)")
    ap.add_argument("--no-track-loop", action="store_true",
                    help="behave like firmware without the track-loop patch (no reply to CMD_TRAJ_NUDGE 0x0B)")
    ap.add_argument("--hover-alt", type=float, default=0.5, help="height the fake flies at (m); 0.8 = like --height 0.8")
    ap.add_argument("--start", default="0,0", help="start x,y in the drone's local frame, e.g. -1.9,-0.2 (like a drone that was not power-cycled)")
    args = ap.parse_args(argv)
    d = FakeDrone(args.drone_id, parse_tags(args.tags) if args.tags else None,
                  mount=CameraMount(pitch_deg=args.camera_pitch_deg), frame_age_ms=args.frame_age_ms,
                  start_pose=(*map(float, args.start.split(",")), 0.0),
                  full_telemetry=args.full_telemetry, mavlink_port=args.mavlink_port,
                  camera_settings=not args.no_camera_settings,
                  glitch=tuple(map(float, args.glitch.split(","))) if args.glitch else None,
                  drift=tuple(map(float, args.drift.split(","))) if args.drift else None,
                  yaw_drift=tuple(map(float, args.yaw_drift.split(","))) if args.yaw_drift else None,
                  takeoff_drift=tuple(map(float, args.takeoff_drift.split(","))) if args.takeoff_drift else None,
                  hold_bias=tuple(map(float, args.hold_bias.split(","))) if args.hold_bias else None,
                  hover_alt=args.hover_alt,
                  nose_bias=tuple(map(float, args.nose_bias.split(","))) if args.nose_bias else None,
                  track_loop=not args.no_track_loop).start()
    print(f"fake drone {args.drone_id} running on 127.0.0.1 (tags {d.tags}); Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        d.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
