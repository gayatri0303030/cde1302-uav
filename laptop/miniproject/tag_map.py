"""Per-tag aggregation of AprilTag observations.

One detection is noisy (pixel noise, oblique views, a stale pose).  The map keeps every
accepted observation per tag ID and reports ONE robust position per ID:

    1. median of north and of east over all observations,
    2. keep only observations within `inlier_radius_m` of that median point,
    3. average the survivors with weight 1 / range^2 (near views are more accurate).

This also deduplicates: the mission reports one position per unique ID, however many
frames saw it.

Accepted observation vs confirmed tag (false-ID guard):
    an observation is ACCEPTED when it passes every quality/pose/range/timing check;
    an ID is CONFIRMED only once accepted observations of it came from `min_frames` DISTINCT
    camera frames (not necessarily consecutive).  A frame is identified by (frame_id, capture
    time), so re-processing the same frame never adds to the count, and a second detection of
    the same ID in one frame is ignored.  Only confirmed IDs belong in the final target list.
"""

from __future__ import annotations

import math
import statistics
import threading
from dataclasses import dataclass, field


@dataclass
class Observation:
    """One tag seen in one camera frame (accepted or rejected)."""
    tag_id: int
    frame_id: int
    t_cap: float                    # estimated capture time, laptop monotonic s
    t_epoch: float                  # same instant as wall-clock epoch seconds
    margin: float
    hamming: int
    range_m: float | None
    tx: float | None                # camera-frame translation (m)
    ty: float | None
    tz: float | None
    cx: float                       # tag centre in the image (px)
    cy: float
    pose_x: float | None = None     # drone pose interpolated at t_cap
    pose_y: float | None = None
    pose_yaw: float | None = None
    fwd: float | None = None        # tag in the body frame (m)
    right: float | None = None
    down: float | None = None
    north: float | None = None      # tag in local NED (m)
    east: float | None = None
    accepted: bool = False
    reason: str = ""                # why it was rejected ("" when accepted)
    # raw extras for raw_detections.csv (no calculation beyond unit conversion)
    corners: tuple | None = None    # 4 x (u, v) px, AprilTag pixel convention, as the detector returned them
    tag_q: tuple | None = None      # tag->camera rotation as quaternion (qx, qy, qz, qw)
    pose_err: float | None = None   # detector's object-space pose error
    reproj_px: float | None = None  # mean distance between detected corners and the pose reprojected
    raw_px4: object | None = None   # nearest RAW telemetry sample (PoseSample), no interpolation
    accepted_frame_count: int = 0   # distinct accepted frames of this ID so far (after this one)
    confirmed_at: float | None = None  # epoch when this ID became confirmed (None = not yet)


@dataclass
class Estimate:
    tag_id: int
    north: float
    east: float
    n_obs: int                      # accepted observations of this tag (one per distinct frame)
    n_inliers: int
    std_north: float
    std_east: float
    best_range_m: float
    first_t_epoch: float
    last_t_epoch: float
    mean_down: float
    accepted_frame_count: int = 0
    confirmed_at: float | None = None


@dataclass
class _Track:
    obs: list = field(default_factory=list)
    frames: set = field(default_factory=set)      # distinct accepted frame keys
    confirmed_at: float | None = None              # epoch of the confirming observation


# return values of TagMap.add()
NEW = "new"                 # first accepted frame of this ID (candidate, not yet confirmed)
COUNTED = "counted"         # another distinct accepted frame, still not confirmed
CONFIRMED = "confirmed"     # this frame confirmed the ID (returned exactly once per mission)
DUPLICATE = "duplicate"     # this frame was already counted for this ID: ignored
IGNORED = ""                # not an accepted observation: nothing stored


def frame_key(obs: Observation) -> tuple:
    return (int(obs.frame_id), round(float(obs.t_cap), 3))


class TagMap:
    def __init__(self, inlier_radius_m: float = 0.20, min_frames: int = 3):
        self.inlier_radius_m = inlier_radius_m
        self.min_frames = max(1, int(min_frames))
        self._tracks: dict[int, _Track] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget every observation, count and confirmation (call at the start of each mission)."""
        with self._lock:
            self._tracks.clear()

    def shift(self, d_north: float, d_east: float) -> None:
        """Move every stored observation by (-d_north, -d_east): used when (0, 0) is re-set
        (e.g. from the home tag after take-off) so earlier sightings use the new zero too."""
        with self._lock:
            for tr in self._tracks.values():
                for o in tr.obs:
                    if o.north is not None and o.east is not None:
                        o.north -= d_north
                        o.east -= d_east

    def add(self, obs: Observation) -> str:
        """Store an ACCEPTED observation once per distinct frame; returns NEW / COUNTED /
        CONFIRMED / DUPLICATE, or IGNORED ("") for a rejected one.  Also fills
        obs.accepted_frame_count and obs.confirmed_at for logging."""
        if not obs.accepted or obs.north is None or obs.east is None:
            with self._lock:
                tr = self._tracks.get(obs.tag_id)
                obs.accepted_frame_count = len(tr.frames) if tr else 0
                obs.confirmed_at = tr.confirmed_at if tr else None
            return IGNORED
        key = frame_key(obs)
        with self._lock:
            tr = self._tracks.get(obs.tag_id)
            if tr is None:
                tr = self._tracks[obs.tag_id] = _Track()
            if key in tr.frames:
                status = DUPLICATE
            else:
                tr.frames.add(key)
                tr.obs.append(obs)
                if tr.confirmed_at is None and len(tr.frames) >= self.min_frames:
                    tr.confirmed_at = obs.t_epoch
                    status = CONFIRMED
                else:
                    status = NEW if len(tr.frames) == 1 else COUNTED
            obs.accepted_frame_count = len(tr.frames)
            obs.confirmed_at = tr.confirmed_at
            return status

    def ids(self) -> list[int]:
        """Every ID with at least one accepted observation (confirmed or not)."""
        with self._lock:
            return sorted(t for t, tr in self._tracks.items() if tr.frames)

    def confirmed_ids(self) -> list[int]:
        with self._lock:
            return sorted(t for t, tr in self._tracks.items() if tr.confirmed_at is not None)

    def is_confirmed(self, tag_id: int) -> bool:
        with self._lock:
            tr = self._tracks.get(tag_id)
            return tr is not None and tr.confirmed_at is not None

    def frame_count(self, tag_id: int) -> int:
        with self._lock:
            tr = self._tracks.get(tag_id)
            return len(tr.frames) if tr else 0

    def confirmed_at(self, tag_id: int) -> float | None:
        with self._lock:
            tr = self._tracks.get(tag_id)
            return tr.confirmed_at if tr else None

    def estimate(self, tag_id: int) -> Estimate | None:
        with self._lock:
            tr = self._tracks.get(tag_id)
            obs = list(tr.obs) if tr else []
        if not obs:
            return None
        med_n = statistics.median(o.north for o in obs)
        med_e = statistics.median(o.east for o in obs)
        inl = [o for o in obs if math.hypot(o.north - med_n, o.east - med_e) <= self.inlier_radius_m]
        if not inl:                                     # all scattered: fall back to the median point
            inl = obs
        w = [1.0 / max(o.range_m or 1.0, 0.1) ** 2 for o in inl]
        sw = sum(w)
        n = sum(wi * o.north for wi, o in zip(w, inl)) / sw
        e = sum(wi * o.east for wi, o in zip(w, inl)) / sw
        sd = (lambda vals: statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        return Estimate(
            tag_id=tag_id, north=n, east=e, n_obs=len(obs), n_inliers=len(inl),
            std_north=sd([o.north for o in inl]), std_east=sd([o.east for o in inl]),
            best_range_m=min(o.range_m for o in obs if o.range_m is not None),
            first_t_epoch=min(o.t_epoch for o in obs), last_t_epoch=max(o.t_epoch for o in obs),
            mean_down=statistics.fmean(o.down for o in obs if o.down is not None),
            accepted_frame_count=len(tr.frames), confirmed_at=tr.confirmed_at,
        )

    def estimates(self) -> list[Estimate]:
        return [e for e in (self.estimate(i) for i in self.ids()) if e is not None]
