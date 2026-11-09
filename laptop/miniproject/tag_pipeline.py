"""Detections -> accepted, localised tag observations.

Pure logic (no sockets, no threads of its own), so it can be unit tested and replayed.
For every detection in a frame:

    1. quality gates      (firmware gate, pose validity, range, image edge)
    2. time sync          (drone pose at the frame's estimated CAPTURE time)
    3. gates on the pose  (telemetry gap, yaw rate)
    4. geometry           (camera -> body -> local NED)
    5. hand the observation to the tag map and the recorder
"""

from __future__ import annotations

import math
import time

from .geometry import localise
from .pose_history import PoseHistory
from .recorder import Recorder
from .settings import Settings
from .tag_map import CONFIRMED, NEW, Observation, TagMap

FRAME_W, FRAME_H = 320, 240        # QVGA, what the firmware streams


def rotation_to_quat(R) -> tuple[float, float, float, float]:
    """Row-major 3x3 rotation (9 numbers) -> unit quaternion (qx, qy, qz, qw), qw >= 0."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return qx, qy, qz, qw


class TagPipeline:
    def __init__(self, settings: Settings, history: PoseHistory, tag_map: TagMap,
                 recorder: Recorder, on_first_detection=None, refs: dict | None = None,
                 on_confirmed=None):
        self.s = settings
        self.history = history
        self.tag_map = tag_map
        self.rec = recorder
        self.on_first = on_first_detection      # first accepted frame of an ID (candidate)
        self.on_confirmed = on_confirmed        # ID reached min_frames distinct accepted frames
        self.refs = refs or {}
        self.origin = None        # (x, y, yaw) of the drone when the run began; None = report in PX4 local frame
        self.frames = 0
        self.rejected_by_reason: dict[str, int] = {}
        self.special_seen: dict[int, int] = {}   # accepted frames of each special (home) tag
        self.special_ids: set[int] = set()     # e.g. the HOME tag: logged + passed to listeners,
                                               # but never a target (not in the tag map / reports)
        self.listeners: list = []              # callables(obs) for every detection

    def set_origin(self, x: float, y: float, yaw: float) -> None:
        """Report tag positions relative to this pose: origin = here, +North = the nose direction here.

        Without it, positions are in PX4's own local frame, which only equals 'measured from the
        takeoff point' if the drone was power-cycled at that point.
        """
        self.origin = (x, y, yaw)

    def _to_origin(self, n: float, e: float) -> tuple[float, float]:
        if self.origin is None:
            return n, e
        x0, y0, yaw0 = self.origin
        dn, de = n - x0, e - y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        return dn * c + de * s, -dn * s + de * c

    # `dets` are apriltag_host.Detection objects (duck-typed: id, hamming, decision_margin,
    # centre, pose_valid, t, pose_err, range_m).
    def process(self, frame_id: int, t_cap: float, dets: list, gate_reason=None) -> list[Observation]:
        """t_cap is the estimated capture time on the laptop monotonic clock."""
        self.frames += 1
        g, tm = self.s.gate, self.s.timing
        epoch = time.time() - (time.monotonic() - t_cap)
        out: list[Observation] = []
        for d in dets:
            t = d.t if d.pose_valid else None
            o = Observation(
                tag_id=int(d.id), frame_id=frame_id, t_cap=t_cap, t_epoch=epoch,
                margin=float(d.decision_margin) if self._finite(d.decision_margin) else float("nan"),
                hamming=int(d.hamming) if self._finite(d.hamming) else -1,
                range_m=d.range_m if t else None,
                tx=t[0] if t else None, ty=t[1] if t else None, tz=t[2] if t else None,
                cx=float(d.centre[0]), cy=float(d.centre[1]))
            o.corners = tuple((float(u), float(v)) for u, v in d.corners)
            o.pose_err = d.pose_err
            if d.pose_valid and d.R is not None and t:
                o.tag_q = rotation_to_quat(d.R)
                o.reproj_px = self._reproj_px(d)
            o.raw_px4 = self.history.nearest(t_cap)
            o.reason = self._reject_reason(d, o, gate_reason)
            if not o.reason:
                got = self.history.at(t_cap, max_gap_s=tm.max_pose_gap_s, edge_tol_s=tm.edge_tol_s)
                if got is None:
                    o.reason = "no_pose_sync"
                else:
                    pose, yaw_rate = got
                    o.pose_x, o.pose_y, o.pose_yaw = pose.x, pose.y, pose.yaw
                    if abs(yaw_rate) > g.max_yaw_rate_rad_s:
                        o.reason = "yaw_rate"
                    else:
                        r = localise(o.tx, o.ty, o.tz, pose.x, pose.y, pose.yaw, self.s.mount)
                        o.fwd, o.right, o.down = r["fwd"], r["right"], r["down"]
                        o.north, o.east = self._to_origin(r["north"], r["east"])
                        o.accepted = True
            if o.reason:
                self.rejected_by_reason[o.reason] = self.rejected_by_reason.get(o.reason, 0) + 1
            if o.tag_id in self.special_ids:
                status = None                   # not a target
                if o.accepted:
                    self.special_seen[o.tag_id] = self.special_seen.get(o.tag_id, 0) + 1
                o.special_seen = self.special_seen.get(o.tag_id, 0)   # for the --show overlay
            else:
                status = self.tag_map.add(o)    # fills o.accepted_frame_count / o.confirmed_at
            self.rec.observation(o)             # every detection is logged, accepted or not
            self.rec.raw_detection(o)
            for fn in self.listeners:
                try:
                    fn(o)
                except Exception:               # a listener must never break detection
                    pass
            if status == NEW and self.on_first:
                self.on_first(o)
            if status == CONFIRMED and self.on_confirmed:
                self.on_confirmed(o)
            out.append(o)
        return out

    def _reproj_px(self, d) -> float | None:
        """Mean pixel distance between the detected corners and the tag pose projected back."""
        from apriltag_host import CAMERA_INTRINSICS as K      # 9 Nov: the calibration
        h = 0.5 * self.s.tag_size_m
        obj = ((-h, h, 0.0), (h, h, 0.0), (h, -h, 0.0), (-h, -h, 0.0))     # same order as d.corners
        R, t = d.R, d.t
        err = 0.0
        for (px, py, pz), (u, v) in zip(obj, d.corners):
            cx = R[0] * px + R[1] * py + R[2] * pz + t[0]
            cy = R[3] * px + R[4] * py + R[5] * pz + t[1]
            cz = R[6] * px + R[7] * py + R[8] * pz + t[2]
            if cz <= 1e-6:
                return None
            err += math.hypot(K.fx * cx / cz + K.cx - u, K.fy * cy / cz + K.cy - v)
        return err / 4.0

    @staticmethod
    def _finite(v) -> bool:
        try:
            return v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    def _reject_reason(self, d, o: Observation, gate_reason=None) -> str:
        """Quality gate first (the firmware's rule), then the existing pose/range/edge checks.

        gate_reason is accepted for backward compatibility but no longer needed: the same rule
        is applied here, plus rejection of missing / non-finite quality values."""
        g = self.s.gate
        ham, mar = getattr(d, "hamming", None), getattr(d, "decision_margin", None)
        if not self._finite(ham):
            return "gate:hamming_missing"
        if not self._finite(mar):
            return "gate:margin_missing"
        known = int(getattr(o, "tag_id", -1)) in tuple(getattr(g, "known_ids", ()) or ())
        lim = max(g.max_hamming, int(getattr(g, "max_hamming_known", g.max_hamming))) if known else g.max_hamming
        if int(ham) > lim:                            # 6 Nov (D21): tag 20 read 6x at the landing, all 1-bit
            return f"gate:hamming>{lim}"               # fixes -> all thrown away; known tags may use 1 bit
        thr = g.min_margin
        h0 = getattr(g, "min_margin_h0", 0.0) or 0.0
        if int(ham) == 0 and 0.0 < h0 < thr:          # perfect decode: a lower contrast is fine
            thr = h0
        if float(mar) <= thr:
            return f"gate:margin<={thr:g}"
        if g.allowed_ids and int(d.id) not in g.allowed_ids:
            return "id_not_allowed"
        if not d.pose_valid:
            return "no_pose"
        if d.pose_err is not None and d.pose_err > g.max_pose_err:
            return "pose_err"
        if o.range_m is not None and o.range_m > g.max_range_m:
            far = getattr(g, "far_range_m", 0.0) or 0.0
            clean = int(ham) == 0 and float(mar) >= getattr(g, "far_min_margin", 100.0)
            if not (far > g.max_range_m and o.range_m <= far and clean):
                return "too_far"
        cx, cy = o.cx, o.cy
        if (cx < g.edge_px or cy < g.edge_px or cx > FRAME_W - g.edge_px or cy > FRAME_H - g.edge_px):
            return "near_edge"
        if not all(math.isfinite(v) for v in (o.tx, o.ty, o.tz)):
            return "bad_pose"
        return ""
