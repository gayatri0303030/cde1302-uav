"""Strict, safe YAML loading and measurement validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import yaml


class ConfigError(ValueError):
    pass


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int)) or key in result:
            raise ConfigError(f"duplicate or invalid YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = yaml.load(handle, Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    return value


TEMPLATE = Path(__file__).resolve().parents[1] / "mapper_config.example.yaml"


def _merge(default, supplied, path=""):
    if not isinstance(supplied, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")
    extra = set(supplied) - set(default)
    if extra:
        raise ConfigError(f"{path}: unknown keys {sorted(extra)}")
    result = deepcopy(default)
    for key, value in supplied.items():
        name = f"{path}.{key}".strip(".")
        base = default[key]
        if isinstance(base, dict):
            result[key] = _merge(base, value, name)
        else:
            if isinstance(base, bool) and not isinstance(value, bool):
                raise ConfigError(f"{name} must be boolean")
            if isinstance(base, list) and not isinstance(value, list):
                raise ConfigError(f"{name} must be a list")
            if isinstance(base, str) and not isinstance(value, str):
                raise ConfigError(f"{name} must be a string")
            if isinstance(base, (float, int)) and not isinstance(base, bool):
                number(value, name)
            result[key] = value
    return result


def number(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ConfigError(f"{name} must be finite numeric")
    if positive and value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def vector(value, length, name, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{name} must contain {length} numbers")
    return np.array([number(v, name, positive) for v in value], float)


def identity(record):
    family, tag_id = record.get("family_or_dictionary"), record.get("tag_id")
    if (
        not isinstance(family, str)
        or not family
        or type(tag_id) is not int
        or tag_id < 0
    ):
        raise ConfigError(
            "tag identity requires family_or_dictionary and nonnegative integer tag_id"
        )
    return family, tag_id


ANCHOR_DEFAULT = dict(
    enabled=False,
    family_or_dictionary="tag16h5",
    tag_id=0,
    frame="ground_start_NED",
    prior_type="position_xy",
    position_m=[None, None, 0.0],
    position_sigma_xy_m=[None, None],
    surveyed_yaw_rad=None,
    surveyed_yaw_sigma_rad=None,
    measurement_notes="",
)
SIZE_DEFAULT = dict(
    family_or_dictionary="tag16h5",
    tag_id=0,
    size_m=0.12,
    size_measured=False,
    size_sigma_m=None,
)


def validate_anchors(records, config):
    if not isinstance(records, list):
        raise ConfigError("known_tags must be a list")
    seen, result = set(), []
    for supplied in records:
        a = _merge(ANCHOR_DEFAULT, supplied, "known_tags")
        key = identity(a)
        if key in seen:
            raise ConfigError(f"duplicate anchor {key}")
        seen.add(key)
        if a["enabled"]:
            if a["frame"] != "ground_start_NED" or a["prior_type"] != "position_xy":
                raise ConfigError("anchors require ground_start_NED position_xy")
            p = vector(a["position_m"], 3, "anchor position_m")
            if abs(p[2]) > config["ground"]["survey_z_tolerance_m"]:
                raise ConfigError("surveyed ground z must be zero")
            vector(a["position_sigma_xy_m"], 2, "anchor sigma", True)
            if (
                key[0] != config["detector"]["family_or_dictionary"]
                or key[1] not in config["detector"]["allowed_tag_ids"]
            ):
                raise ConfigError(
                    f"anchor {key} outside detector whitelist; update it explicitly"
                )
            if a["surveyed_yaw_rad"] is not None:
                number(a["surveyed_yaw_rad"], "surveyed_yaw_rad")
                number(a["surveyed_yaw_sigma_rad"], "surveyed_yaw_sigma_rad", True)
            elif a["surveyed_yaw_sigma_rad"] is not None:
                raise ConfigError("yaw sigma without surveyed yaw")
        result.append(a)
    return result


def load_config(path, anchors_path=None):
    defaults = read_yaml(TEMPLATE)
    c = _merge(defaults, read_yaml(path))
    # These describe the implemented model, not interchangeable algorithms.
    fixed = {
        "schema_version": 2,
        "units.length": "m",
        "units.angle": "rad",
        "units.internal_time": "ns",
        "frames.telemetry_navigation": "NED",
        "frames.world_output": "ground_start_NED",
        "frames.physical_body": "FRD",
        "frames.internal_quaternion_order": "xyzw",
        "frames.drone_navigation_output": "drone_centered_NED",
        "frames.camera": "opencv_optical_right_down_forward",
        "frames.transform_convention": "T_A_B_maps_B_to_A_column_vectors",
        "frames.internal_attitude_direction": "body_to_navigation",
        "frames.start.mode": "explicit_ground_reference_event",
        "frames.start.axes": "NED",
        "frames.start.subtract_initial_heading": False,
        "ground.world_z_m": 0.0,
        "ground.optimize_tag_z": False,
        "ground.optimize_tag_roll_pitch": False,
        "ground.constraint": "hard_by_state_parameterization",
        "ground.all_tags_flat_and_static": True,
        "tag_geometry.ground_state_components": ["x_m", "y_m", "theta_rad"],
        "tag_geometry.static_landmarks": True,
        "camera.model": "opencv_pinhole",
        "camera.pixels": "raw_distorted",
        "camera.distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "initialization.nonlinear_pnp_refinement": False,
        "initialization.use_tag_size_for_ray_depth": False,
        "initialization.use_tag_size_for_graph_corner_geometry": True,
        "initialization.require_forward_ground_intersection": True,
        "detector.adapter": "apriltag_c_host_shim",
        "detector.family_or_dictionary": "tag16h5",
        "detector.corner_contract": "apriltag_det_p0_p1_p2_p3",
        "mavlink.wire_quaternion_order": "wxyz",
        "mavlink.apply_display_repr_offset": False,
        "optimization.solver": "gauss_newton",
        "optimization.backend": "explicit_python_gauss_newton",
        "optimization.mode": "offline_batch",
        "optimization.optimize_tag_z": False,
        "optimization.tag_state": "planar_x_y_theta",
        "optimization.linear_damping": 0.0,
        "optimization.optimize_camera_extrinsic": False,
        "optimization.optimize_intrinsics": False,
        "optimization.optimize_tag_size": False,
        "optimization.optimize_time_offset": False,
        "optimization.pose_update": "right_SE3_retraction",
        "optimization.robust_loss": "huber_block_irls",
        "optimization.absolute_px4_prior_at_every_pose": False,
        "optimization.silent_solver_substitution": False,
        "optimization.explicit_matrix_inverse": False,
        "optimization.visual_factor_mode": "raw_corner_reprojection",
        "optimization.line_search.require_actual_robust_cost_decrease": True,
        "optimization.line_search.require_positive_observed_corner_depth": True,
        "optimization.pose_tangent_order": [
            "rx_rad",
            "ry_rad",
            "rz_rad",
            "tx_m",
            "ty_m",
            "tz_m",
        ],
        "output.publish_to_flight_controller": False,
        "output.tag_world_z_m": 0.0,
        "timing.position_interpolation": "linear",
        "timing.attitude_interpolation": "shortest_arc_slerp",
        "timing.max_extrapolation_ms": 0,
        "evaluation.load_survey_only_after_fit": True,
        "evaluation.align_to_held_out_truth": False,
        "evaluation.do_not_use_for_initialization_or_weight_tuning": True,
        "output.immutable_raw_logs": True,
        "output.log_no_detection_frames": True,
    }
    for name, expected in fixed.items():
        actual = c
        for part in name.split("."):
            actual = actual[part]
        if actual != expected:
            raise ConfigError(f"{name}: supported value is {expected!r}")

    def finite_tree(value, name="config"):
        if isinstance(value, dict):
            for k, v in value.items():
                finite_tree(v, f"{name}.{k}")
        elif isinstance(value, list):
            for v in value:
                finite_tree(v, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number(value, name)

    finite_tree(c)
    for name in ("optimize_drone_trajectory", "optimize_tag_xy_yaw"):
        if not c["optimization"][name]:
            raise ConfigError(f"optimization.{name} must be true")
    processing = c["camera"]["image_processing"]
    if processing != defaults["camera"]["image_processing"]:
        raise ConfigError(
            "use calibration matching raw images; image transforms are not supported in v1"
        )
    cam = c["camera"]
    k = np.asarray(cam["K"], float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or not np.allclose(k[2], [0, 0, 1])
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or k[0, 1] != 0
        or k[1, 0] != 0
    ):
        raise ConfigError("invalid pinhole K")
    vector(cam["distortion"], 5, "distortion")
    for name in ("width", "height"):
        if type(cam[name]) is not int or cam[name] <= 0:
            raise ConfigError(f"camera.{name} must be a positive integer")
    mount = cam["extrinsic_T_B_C"]
    r = np.asarray(mount["rotation_matrix"], float)
    if (
        r.shape != (3, 3)
        or not np.isfinite(r).all()
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(r), 1)
    ):
        raise ConfigError("mount rotation must be SO(3)")
    vector(mount["translation_m"], 3, "mount translation")
    number(c["tag_geometry"]["default_size_m"], "tag size", True)
    if c["tag_geometry"]["default_size_sigma_m"] is not None:
        number(c["tag_geometry"]["default_size_sigma_m"], "default_size_sigma_m", True)
    for key in ("translation_sigma_m", "rotation_sigma_rad"):
        if mount[key] is not None:
            vector(mount[key], 3, key, True)
    seen = set()
    for s in c["tag_geometry"]["per_tag"]:
        if not isinstance(s, dict):
            raise ConfigError("per_tag entries must be mappings")
        s.update(_merge(SIZE_DEFAULT, s, "per_tag"))
        key = identity(s)
        if key in seen:
            raise ConfigError(f"duplicate size override {key}")
        seen.add(key)
        number(s["size_m"], "size_m", True)
        if s["size_sigma_m"] is not None:
            number(s["size_sigma_m"], "size_sigma_m", True)
    if anchors_path:
        if any(isinstance(a, dict) and a.get("enabled") for a in c["known_tags"]):
            raise ConfigError(
                "external anchors conflict with enabled inline known_tags"
            )
        external = read_yaml(anchors_path)
        if set(external) != {"known_tags"}:
            raise ConfigError("anchor YAML must contain only known_tags")
        c["known_tags"] = external["known_tags"]
    c["known_tags"] = validate_anchors(c["known_tags"], c)
    u, o = c["uncertainty"], c["optimization"]
    for name in ("corner_sigma_px", "fallback_sigma_reference_interval_s"):
        number(u[name], name, True)
    for name in (
        "fallback_odometry_rotation_sigma_rad",
        "fallback_odometry_translation_sigma_m",
        "start_attitude_prior_sigma_rad",
    ):
        vector(u[name], 3, name, True)
    for name in (
        "huber_delta_whitened",
        "maximum_iterations",
        "relative_cost_tolerance",
        "translation_step_tolerance_m",
        "rotation_step_tolerance_rad",
        "gradient_infinity_tolerance",
    ):
        number(o[name], name, True)
    if type(o["maximum_iterations"]) is not int:
        raise ConfigError("maximum_iterations must be integer")
    if o["linear_solver"] not in ("sparse_lsmr_or_sparse_qr", "sparse_lsmr"):
        raise ConfigError("only the sparse_lsmr linear backend is implemented")
    if not set(o["robustify_factor_types"]) <= {
        "pixel",
        "motion",
        "anchor",
        "yaw",
        "height",
        "attitude",
    }:
        raise ConfigError("unknown robust factor type")
    for name in (
        "ground.survey_z_tolerance_m",
        "gating.minimum_apriltag_decision_margin",
    ):
        section, key = name.split(".")
        number(c[section][key], name, True)
    line = o["line_search"]
    if (
        not line["enabled"]
        or not 0 < line["shrink_factor"] < 1
        or not 0 < line["minimum_alpha"] <= line["initial_alpha"] <= 1
    ):
        raise ConfigError("invalid line search")
    number(c["timing"]["max_bracket_gap_ms"], "max_bracket_gap_ms", True)
    for name in ("scale", "sync_sigma_ns"):
        if c["timing"][name] is not None:
            number(c["timing"][name], name, True)
    for name in ("offset_ns", "capture_timestamp_offset_ns"):
        if c["timing"][name] is not None and type(c["timing"][name]) is not int:
            raise ConfigError(f"timing.{name} must be integer nanoseconds")
    for name in ("initial_body_height_above_floor_m", "initial_body_height_sigma_m"):
        if c["frames"]["start"][name] is not None:
            number(c["frames"]["start"][name], name, True)
    if c["frames"]["start"]["native_ground_origin_position_m"] is not None:
        vector(
            c["frames"]["start"]["native_ground_origin_position_m"], 3, "ground origin"
        )
    ids = c["detector"]["allowed_tag_ids"]
    if any(type(i) is not int or not 0 <= i < 30 for i in ids) or len(set(ids)) != len(
        ids
    ):
        raise ConfigError(
            "allowed_tag_ids must be unique IDs in the compiled tag16h5 table (0..29)"
        )
    g = c["gating"]
    for name in (
        "minimum_quad_area_px2",
        "maximum_corner_euclidean_rms_px",
        "minimum_positive_corner_depth_m",
    ):
        number(g[name], name, True)
    if (
        type(g["maximum_hamming_distance"]) is not int
        or g["maximum_hamming_distance"] < 0
    ):
        raise ConfigError("maximum_hamming_distance must be nonnegative integer")
    if not 0 < c["initialization"]["minimum_unit_ray_down_component"] <= 1:
        raise ConfigError("minimum ray down component must lie in (0,1]")
    number(o["keyframes"]["minimum_interval_s"], "keyframe interval", True)
    legacy = c["legacy_replay"]
    number(legacy["assumed_height_above_floor_m"], "assumed height", True)
    if legacy["assumed_drone_world_z_m"] != -legacy["assumed_height_above_floor_m"]:
        raise ConfigError("legacy world z must be negative height above floor")
    if c["mavlink"]["primary_pose_source"] not in (
        "LOCAL_POSITION_NED_plus_ATTITUDE_QUATERNION",
        "ODOMETRY",
    ):
        raise ConfigError("unsupported primary_pose_source")
    return c


def unverified(c):
    checks = [
        (c["frames"]["navigation_to_true_ned_verified"], "NED alignment"),
        (c["camera"]["extrinsic_T_B_C"]["measurement_verified"], "camera mount"),
        (c["camera"]["held_out_calibration_validation_complete"], "camera validation"),
        (c["detector"]["preserves_native_corner_order_verified"], "corner order"),
        (c["tag_geometry"]["default_size_measured"], "tag size"),
    ]
    missing = [name for ok, name in checks if not ok]
    for key in ("translation_sigma_m", "rotation_sigma_rad"):
        if c["camera"]["extrinsic_T_B_C"][key] is None:
            missing.append("mount " + key)
    if c["tag_geometry"]["default_size_sigma_m"] is None:
        missing.append("tag size uncertainty")
    for s in c["tag_geometry"]["per_tag"]:
        if not s["size_measured"] or s["size_sigma_m"] is None:
            missing.append(f"unverified per-tag size {identity(s)}")
    return missing


def tag_size(c, key):
    return next(
        (s["size_m"] for s in c["tag_geometry"]["per_tag"] if identity(s) == key),
        c["tag_geometry"]["default_size_m"],
    )
