"""Run from laptop/:  python3 -m pytest tests -q"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from miniproject.geometry import CameraMount, body_to_cam, cam_to_body, localise, local_to_body, body_to_local, wrap_pi
from miniproject.pose_history import PoseHistory
from miniproject.tag_map import Observation, TagMap
from miniproject.settings import load_settings


def test_cam_body_roundtrip():
    m = CameraMount(pitch_deg=45, offset_fwd_m=0.02)
    for body in [(0.6, 0.1, 0.5), (0.3, -0.2, 0.5)]:
        assert cam_to_body(*body_to_cam(*body, m), m) == pytest.approx(body, abs=1e-9)


def test_body_local_roundtrip():
    for yaw in (0.0, 1.0, -2.5, math.pi):
        n, e = body_to_local(0.4, -0.2, 1.0, 2.0, yaw)
        assert local_to_body(n, e, 1.0, 2.0, yaw) == pytest.approx((0.4, -0.2), abs=1e-9)


def test_localise_floor_tag_height():
    m = CameraMount(pitch_deg=45)
    tx, ty, tz = body_to_cam(0.6, 0.1, 0.5, m)
    r = localise(tx, ty, tz, 1.0, 2.0, 0.0, m)
    assert r["down"] == pytest.approx(0.5, abs=1e-9)
    assert (r["north"], r["east"]) == pytest.approx((1.6, 2.1), abs=1e-9)


def test_pose_history_interpolates_and_wraps():
    h = PoseHistory()
    h.add(0.0, 0.0, 0.0, math.pi - 0.1)
    h.add(0.2, 1.0, 0.0, -math.pi + 0.1)
    p, _ = h.at(0.1)
    assert p.x == pytest.approx(0.5)
    assert abs(wrap_pi(p.yaw - math.pi)) < 1e-6       # went the short way round
    assert h.at(5.0) is None


def _obs(n, e, rng=0.5, i=0):
    return Observation(tag_id=7, frame_id=i, t_cap=0, t_epoch=i, margin=80, hamming=0, range_m=rng,
                       tx=0, ty=0, tz=rng, cx=160, cy=120, north=n, east=e, down=0.5, accepted=True)


def test_tag_map_rejects_outlier():
    tm = TagMap(0.2)
    for i in range(9):
        tm.add(_obs(1.0 + 0.01 * (i % 3), 2.0, i=i))
    tm.add(_obs(3.0, 3.0, i=99))                       # wild outlier
    e = tm.estimate(7)
    assert e.north == pytest.approx(1.01, abs=0.02) and e.east == pytest.approx(2.0, abs=0.02)
    assert e.n_obs == 10 and e.n_inliers == 9


def test_default_config_loads():
    s = load_settings()
    assert 0 < s.mount.pitch_deg < 90 and s.min_obs >= 2 and s.flight.max_speed_mps <= 0.30
    assert s.gate.max_hamming == 1 and s.gate.min_margin == 55 and s.gate.allowed_ids == ()


def test_origin_shift_and_rotation():
    from miniproject.tag_pipeline import TagPipeline
    from miniproject.recorder import Recorder
    import tempfile
    p = TagPipeline(load_settings(), PoseHistory(), TagMap(0.2), Recorder(tempfile.mkdtemp()))
    assert p._to_origin(1.0, 2.0) == (1.0, 2.0)                      # no origin: PX4 local frame
    p.set_origin(-1.0, 0.5, 0.0)
    assert p._to_origin(0.5, 0.5) == pytest.approx((1.5, 0.0))
    p.set_origin(0.0, 0.0, math.pi / 2)                              # nose points East
    assert p._to_origin(0.0, 1.0) == pytest.approx((1.0, 0.0))       # a tag East of us is straight ahead


# ---- search planner -------------------------------------------------------------------------
def test_planner_stays_inside_and_limits():
    import numpy as np
    from miniproject.planner import Area, SearchParams, plan_search, peak_speed
    area = Area(-0.5, 2.0, -1.0, 1.0)
    pl = plan_search(area, SearchParams(speed_mps=0.2))
    assert pl.xy[:, 0].min() >= area.xmin and pl.xy[:, 0].max() <= area.xmax
    assert pl.xy[:, 1].min() >= area.ymin and pl.xy[:, 1].max() <= area.ymax
    assert peak_speed(pl) <= 0.2 + 1e-6 and len(pl.t) <= 2400
    assert np.allclose(pl.xy[0], 0, atol=1e-6) and np.allclose(pl.xy[-1], 0, atol=1e-3)
    assert pl.coverage > 0.7


def test_planner_rejects_bad_requests():
    import pytest
    from miniproject.planner import Area, SearchParams, plan_search
    with pytest.raises(ValueError):
        plan_search(Area(0, 2, -1, 1), SearchParams(speed_mps=0.5))
    with pytest.raises(ValueError):
        plan_search(Area(0, 0.5, -0.2, 0.2), SearchParams())


def test_footprint_reach():
    from miniproject.footprint import reach_ahead
    from miniproject.geometry import CameraMount
    near, far = reach_ahead(0.5, CameraMount())
    assert 0.0 < near < far < 1.0


def test_rotation_to_quat_identity_and_z90():
    import math
    from miniproject.tag_pipeline import rotation_to_quat
    assert rotation_to_quat((1, 0, 0, 0, 1, 0, 0, 0, 1)) == (0.0, 0.0, 0.0, 1.0)
    c = math.cos(math.pi / 4)
    q = rotation_to_quat((0, -1, 0, 1, 0, 0, 0, 0, 1))      # +90 deg about z
    assert abs(q[2] - c) < 1e-9 and abs(q[3] - c) < 1e-9


# ---- AprilTag filtering: accepted observation vs confirmed tag ------------------------------
class _Det:
    """Duck-typed apriltag_host.Detection: a valid tag 0.6 m in front of the camera."""
    def __init__(self, tid=1, hamming=0, margin=200.0):
        self.id, self.hamming, self.decision_margin = tid, hamming, margin
        self.centre = (160.0, 120.0)
        self.corners = ((150.0, 130.0), (170.0, 130.0), (170.0, 110.0), (150.0, 110.0))
        self.pose_valid, self.pose_err = True, 0.01
        self.t = (0.0, 0.0, 0.6)
        self.R = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.range_m = 0.6


def _pipeline(**gate):
    import dataclasses, tempfile
    from miniproject.tag_pipeline import TagPipeline
    from miniproject.recorder import Recorder
    s = load_settings()
    if gate:
        s = dataclasses.replace(s, gate=dataclasses.replace(s.gate, **gate))
    h = PoseHistory()
    for i in range(0, 201):                          # drone hovering still, 10 Hz, t = 0 .. 20 s
        h.add(i * 0.1, 0.0, 0.0, 0.0)
    tm = TagMap(s.inlier_radius_m, min_frames=s.min_obs)
    events = {"new": [], "confirmed": []}
    p = TagPipeline(s, h, tm, Recorder(tempfile.mkdtemp()),
                    on_first_detection=lambda o: events["new"].append(o.tag_id),
                    on_confirmed=lambda o: events["confirmed"].append(o.tag_id))
    return p, tm, events


def test_hamming_2_rejected_hamming_1_accepted():
    p, tm, _ = _pipeline()
    o2 = p.process(1, 5.0, [_Det(hamming=2)])[0]
    o1 = p.process(2, 5.5, [_Det(hamming=1)])[0]
    assert not o2.accepted and o2.reason == "gate:hamming>1"
    assert o1.accepted and o1.reason == ""


def test_margin_55_rejected_and_bad_values_rejected():
    p, tm, _ = _pipeline()
    o55 = p.process(1, 5.0, [_Det(margin=55.0)])[0]
    o_ok = p.process(2, 5.5, [_Det(margin=55.01)])[0]
    o_nan = p.process(3, 6.0, [_Det(margin=float("nan"))])[0]
    o_none = p.process(4, 6.5, [_Det(margin=None)])[0]
    o_inf_h = p.process(5, 7.0, [_Det(hamming=float("inf"))])[0]
    assert not o55.accepted and o55.reason == "gate:margin<=55"
    assert o_ok.accepted
    assert o_nan.reason == "gate:margin_missing" and o_none.reason == "gate:margin_missing"
    assert o_inf_h.reason == "gate:hamming_missing"


def test_two_frames_unconfirmed_three_confirm_once():
    p, tm, ev = _pipeline()
    p.process(10, 5.0, [_Det(tid=3)])
    p.process(12, 6.0, [_Det(tid=3)])                # not consecutive: still counts
    assert tm.frame_count(3) == 2 and not tm.is_confirmed(3) and tm.confirmed_ids() == []
    assert ev["new"] == [3] and ev["confirmed"] == []
    o = p.process(15, 7.0, [_Det(tid=3)])[0]
    assert tm.is_confirmed(3) and tm.confirmed_ids() == [3]
    assert o.accepted_frame_count == 3 and o.confirmed_at is not None
    p.process(16, 7.5, [_Det(tid=3)])                # 4th frame: no second confirmation
    assert ev["confirmed"] == [3]


def test_duplicate_frame_counts_once():
    p, tm, ev = _pipeline()
    p.process(20, 5.0, [_Det(tid=4)])
    p.process(20, 5.0, [_Det(tid=4)])                # same frame re-processed
    p.process(21, 5.5, [_Det(tid=4), _Det(tid=4)])   # same ID twice in one frame
    assert tm.frame_count(4) == 2 and not tm.is_confirmed(4)
    assert tm.estimate(4).n_obs == 2                 # duplicates are not in the position estimate either
    assert ev["confirmed"] == []


def test_rejected_never_counted_but_logged_and_reset():
    import csv
    p, tm, ev = _pipeline()
    for i in range(5):
        p.process(30 + i, 5.0 + i * 0.5, [_Det(tid=6, hamming=2)])     # all rejected
    assert tm.frame_count(6) == 0 and tm.ids() == [] and tm.estimate(6) is None
    rows = list(csv.DictReader(open(p.rec.out_dir / "raw_detections.csv")))
    assert len(rows) == 5 and all(r["accepted"] == "0" and r["reject_reason"] for r in rows)
    assert "accepted_frame_count" in rows[0] and "confirmed_at_time" in rows[0]
    for i in range(3):
        p.process(40 + i, 8.0 + i * 0.5, [_Det(tid=6)])
    assert tm.is_confirmed(6)
    tm.reset()                                       # new mission
    assert tm.ids() == [] and tm.confirmed_ids() == [] and tm.frame_count(6) == 0


def test_allowed_ids_optional():
    p, tm, _ = _pipeline(allowed_ids=(0, 1, 2, 3))
    assert p.process(1, 5.0, [_Det(tid=8)])[0].reason == "id_not_allowed"
    assert p.process(2, 5.5, [_Det(tid=2)])[0].accepted
