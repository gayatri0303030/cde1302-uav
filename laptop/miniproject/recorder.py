"""Mission event log: console lines for the examiners + files for the report.

Console (and events.jsonl) get one line per KEY event with a wall-clock timestamp, so
the examiner can verify them on screen (mini-project brief, section 4.1):

    [10:22:46.123] T+  0.00 s | TAKEOFF_COMMAND_SENT | drone 5
    [10:22:58.502] T+ 12.38 s | TAG_DETECTED | id 3  N +1.234  E -0.405  range 0.88 m  frame 417
    [10:23:31.870] T+ 45.75 s | LAND_COMMAND_SENT | ...

observations.csv gets EVERY detection (accepted or rejected, with the reason), which is
the raw material for the localisation analysis and for fusing with the PX4 .ulg log.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .tag_map import Observation

OBS_COLUMNS = [
    "t_epoch", "t_since_t0_s", "frame_id", "tag_id", "accepted", "reason",
    "margin", "hamming", "range_m", "tx", "ty", "tz", "cx", "cy",
    "pose_x", "pose_y", "pose_yaw_rad", "fwd_m", "right_m", "down_m", "north_m", "east_m",
    "accepted_frame_count", "confirmed_at_time",
]


RAW_COLUMNS = (
    ["px4_timestamp", "px4_x_m", "px4_y_m", "px4_z_m",
     "px4_qx", "px4_qy", "px4_qz", "px4_qw",
     "px4_vx_mps", "px4_vy_mps", "px4_vz_mps",
     "px4_roll_rate_radps", "px4_pitch_rate_radps", "px4_yaw_rate_radps",
     "camera_capture_timestamp", "tag_id"]
    + [f"corner_{i}_{a}" for i in range(4) for a in ("u", "v")]
    + ["decision_margin", "hamming_distance", "tag_tx_m", "tag_ty_m", "tag_tz_m",
       "tag_qx", "tag_qy", "tag_qz", "tag_qw", "reprojection_error_px",
       "frame_id", "accepted", "reject_reason", "px4_yaw_rad", "px4_sample_offset_s",
       "px4_pos_boot_ms", "px4_att_boot_ms", "px4_quat_source",
       "mav_x_m", "mav_y_m", "mav_pos_timestamp", "mav_att_timestamp", "mav_roll_rad", "mav_pitch_rad",
       "accepted_frame_count", "confirmed_at_time"]
)
TELEM_COLUMNS = ["t_epoch_rx", "t_epoch_sample", "px4_x_m", "px4_y_m", "px4_z_m", "px4_yaw_rad",
                 "px4_vx_mps", "px4_vy_mps", "px4_vz_mps", "px4_pos_age_ms", "px4_att_age_ms",
                 "px4_armed", "nav_state",
                 "px4_qw", "px4_qx", "px4_qy", "px4_qz",
                 "px4_roll_rate_radps", "px4_pitch_rate_radps", "px4_yaw_rate_radps",
                 "px4_roll_rad", "px4_pitch_rad", "px4_pos_boot_ms", "px4_att_boot_ms", "px4_q_from_px4"]


def _fmt_clock(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S.%f")[:-3]


class Recorder:
    def __init__(self, out_dir: str | Path, quiet_observations: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._t0: float | None = None            # epoch of the takeoff command / start
        self._events = open(self.out_dir / "events.jsonl", "w", encoding="utf-8")
        self._obs_file = open(self.out_dir / "observations.csv", "w", newline="", encoding="utf-8")
        self._obs_csv = csv.writer(self._obs_file)
        self._obs_csv.writerow(OBS_COLUMNS)
        self._quiet_obs = quiet_observations
        self._raw_file = open(self.out_dir / "raw_detections.csv", "w", newline="", encoding="utf-8")
        self._raw_csv = csv.writer(self._raw_file)
        self._raw_csv.writerow(RAW_COLUMNS)
        self._tel_file = open(self.out_dir / "px4_telemetry.csv", "w", newline="", encoding="utf-8")
        self._tel_csv = csv.writer(self._tel_file)
        self._tel_csv.writerow(TELEM_COLUMNS)

    # ---- clock ---------------------------------------------------------------------
    def set_t0(self, epoch: float | None = None) -> float:
        self._t0 = time.time() if epoch is None else epoch
        return self._t0

    @property
    def t0(self) -> float | None:
        return self._t0

    def since_t0(self, epoch: float) -> float | None:
        return None if self._t0 is None else epoch - self._t0

    # ---- events --------------------------------------------------------------------
    def event(self, name: str, text: str = "", **fields) -> float:
        now = time.time()
        rel = self.since_t0(now)
        rel_s = "T+ ---.-- s" if rel is None else f"T+{rel:7.2f} s"
        line = f"[{_fmt_clock(now)}] {rel_s} | {name}" + (f" | {text}" if text else "")
        with self._lock:
            print(line, flush=True)
            self._events.write(json.dumps({"t_epoch": now, "t_since_t0_s": rel, "event": name,
                                           "text": text, **fields}) + "\n")
            self._events.flush()
        return now

    def note(self, text: str) -> None:
        """Console-only status line (not an official event)."""
        with self._lock:
            print(f"[{_fmt_clock(time.time())}]              | {text}", flush=True)

    # ---- observations --------------------------------------------------------------
    def observation(self, o: Observation) -> None:
        rel = self.since_t0(o.t_epoch)
        f = lambda v, n=4: "" if v is None else round(v, n)
        with self._lock:
            self._obs_csv.writerow([
                f"{o.t_epoch:.3f}", "" if rel is None else f"{rel:.3f}", o.frame_id, o.tag_id,
                int(o.accepted), o.reason, f(o.margin, 1), o.hamming, f(o.range_m), f(o.tx), f(o.ty),
                f(o.tz), f(o.cx, 1), f(o.cy, 1), f(o.pose_x), f(o.pose_y), f(o.pose_yaw, 4),
                f(o.fwd), f(o.right), f(o.down), f(o.north), f(o.east),
                o.accepted_frame_count, "" if o.confirmed_at is None else f"{o.confirmed_at:.3f}"])
            self._obs_file.flush()

    def raw_detection(self, o: Observation) -> None:
        """One row per detection with raw readings only (see RAW_COLUMNS)."""
        r = o.raw_px4
        g = lambda v, n=6: "" if v is None else round(float(v), n)
        c = o.corners or ()
        corner_vals = []
        for i in range(4):
            corner_vals += [g(c[i][0], 3), g(c[i][1], 3)] if i < len(c) else ["", ""]
        q = o.tag_q or (None,) * 4
        pq = (r.q if r is not None and r.q else None) or (None,) * 4          # w, x, y, z
        pr = (r.rates if r is not None and r.rates else None) or (None,) * 3
        qsrc = "" if r is None or r.q_from_px4 is None else (
            "ATTITUDE_QUATERNION" if r.q_from_px4 else "ATTITUDE(euler->quat, on ESP32)")
        z = r.z if r else None
        v3 = (r.vx, r.vy, r.vz) if r else (None, None, None)
        mav = getattr(self, "mav", None)
        mx = my = mpt = mat = mroll = mpitch = None
        if mav is not None and mav.ok:
            # PX4's own MAVLink stream: nearest RAW sample to the camera capture time (no interpolation)
            pos, tp = mav.nearest("LOCAL_POSITION_NED", o.t_epoch)
            if pos is not None:
                z, v3 = pos["z_m"], (pos["vx_mps"], pos["vy_mps"], pos["vz_mps"])
                mx, my, mpt = pos["x_m"], pos["y_m"], tp
            aq, ta = mav.nearest("ATTITUDE_QUATERNION", o.t_epoch)
            at, tb = mav.nearest("ATTITUDE", o.t_epoch)
            if aq is not None:
                pq = (aq["qw"], aq["qx"], aq["qy"], aq["qz"])
                pr = (aq["roll_rate_radps"], aq["pitch_rate_radps"], aq["yaw_rate_radps"])
                qsrc, mat = "MAVLink ATTITUDE_QUATERNION", ta
            elif at is not None:
                import math
                ro, pi, ya = (float(at[k]) for k in ("roll_rad", "pitch_rad", "yaw_rad"))
                cr, sr = math.cos(ro / 2), math.sin(ro / 2)
                cp, sp = math.cos(pi / 2), math.sin(pi / 2)
                cy, sy = math.cos(ya / 2), math.sin(ya / 2)
                pq = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
                pr = (at["roll_rate_radps"], at["pitch_rate_radps"], at["yaw_rate_radps"])
                qsrc, mat = "MAVLink ATTITUDE (euler->quat)", tb
            if at is not None:
                mroll, mpitch = at["roll_rad"], at["pitch_rad"]
        row = [
            "" if r is None or r.epoch is None else f"{r.epoch:.4f}",
            g(r.x if r else None, 4), g(r.y if r else None, 4), g(z, 4),
            g(pq[1]), g(pq[2]), g(pq[3]), g(pq[0]),           # PX4 quaternion qx, qy, qz, qw (empty with old firmware)
            g(v3[0], 4), g(v3[1], 4), g(v3[2], 4),
            g(pr[0]), g(pr[1]), g(pr[2]),                     # PX4 body rates (empty with old firmware)
            f"{o.t_epoch:.4f}", o.tag_id, *corner_vals,
            g(o.margin, 2), o.hamming, g(o.tx, 5), g(o.ty, 5), g(o.tz, 5),
            g(q[0]), g(q[1]), g(q[2]), g(q[3]), g(o.reproj_px, 3),
            o.frame_id, int(o.accepted), o.reason, g(r.yaw if r else None, 5),
            "" if r is None or r.epoch is None else round(o.t_epoch - r.epoch, 4),
            "" if r is None or r.pos_boot_ms is None else r.pos_boot_ms,
            "" if r is None or r.att_boot_ms is None else r.att_boot_ms,
            qsrc,
            g(mx, 4), g(my, 4), "" if mpt is None else f"{mpt:.4f}", "" if mat is None else f"{mat:.4f}",
            g(mroll), g(mpitch),
            o.accepted_frame_count, "" if o.confirmed_at is None else f"{o.confirmed_at:.4f}",
        ]
        with self._lock:
            self._raw_csv.writerow(row)
            self._raw_file.flush()

    def telemetry(self, row: list) -> None:
        with self._lock:
            self._tel_csv.writerow(row)
            self._tel_file.flush()

    # ---- end of run ----------------------------------------------------------------
    def write_json(self, name: str, data) -> Path:
        p = self.out_dir / name
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def close(self) -> None:
        with self._lock:
            for fh in (self._events, self._obs_file, self._raw_file, self._tel_file):
                try:
                    fh.close()
                except OSError:
                    pass
