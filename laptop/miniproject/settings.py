"""Loads config.yaml into typed settings.  Every tunable number lives in config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .geometry import CameraMount

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class Gate:
    max_range_m: float = 1.5
    far_range_m: float = 0.0        # >max_range_m: also accept up to here, but only clean decodes
    far_min_margin: float = 100.0   # (hamming 0 and decision margin >= this); 0 = off
    edge_px: float = 8.0
    max_pose_err: float = 0.5
    max_yaw_rate_rad_s: float = 0.6
    min_margin: float = 55.0        # reject decision_margin <= this (and missing / non-finite)
    min_margin_h0: float = 0.0      # >0: a PERFECT decode (hamming 0) needs only margin > this
    max_hamming: int = 1            # reject hamming > this (and missing / non-finite)
    known_ids: tuple = ()           # 6 Nov: home tag + anchors (KNOWN spots): these may use max_hamming_known
    max_hamming_known: int = 1
    allowed_ids: tuple = ()         # empty = any ID; set on the day if the IDs are announced


@dataclass(frozen=True)
class Timing:
    capture_correction_s: float = 0.0
    pose_latency_s: float = 0.03
    max_pose_gap_s: float = 0.35
    edge_tol_s: float = 0.10


@dataclass(frozen=True)
class Flight:
    takeoff_wait_s: float = 12.0
    max_speed_mps: float = 0.30
    dt_s: float = 0.05
    post_land_observe_s: float = 8.0
    stale_telemetry_s: float = 3.0
    connect_timeout_s: float = 30.0
    expected_altitude_m: float = 0.5
    guard_enabled: bool = True          # land on a PX4 position jump / take-off drift / leaving the path
    max_jump_m: float = 1.0
    max_jump_speed_mps: float = 1.2
    takeoff_max_drift_m: float = 1.0
    max_path_error_m: float = 1.2       # 0 = no path check
    path_error_samples: int = 2


@dataclass(frozen=True)
class Settings:
    tag_size_m: float
    mount: CameraMount
    gate: Gate
    timing: Timing
    flight: Flight
    inlier_radius_m: float
    min_obs: int
    camera_fps: int
    camera_quality: int
    search: dict
    camera_sensor: dict = field(default_factory=dict)   # camera chip settings to apply (None = leave)


def _gate(g: dict) -> Gate:
    g = dict(g)
    g["allowed_ids"] = tuple(int(i) for i in (g.get("allowed_ids") or ()))
    return Gate(**g)


def parse_overrides(items) -> list[tuple[str, object]]:
    """['gate.max_range_m=1.5', 'estimate.min_obs=2'] -> [(dotted key, value)], values typed by YAML."""
    out = []
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"--set {it!r}: write it as section.key=value, e.g. gate.max_range_m=1.5")
        k, v = it.split("=", 1)
        out.append((k.strip(), yaml.safe_load(v.strip()) if v.strip() else None))
    return out


def apply_overrides(raw: dict, overrides) -> None:
    """Set dotted keys in the raw config; only keys that already exist (typos are refused)."""
    for key, val in overrides:
        node, parts = raw, key.split(".")
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise ValueError(f"--set {key}: there is no '{p}' section in config.yaml")
            node = node[p]
        if not isinstance(node, dict) or parts[-1] not in node:
            known = ", ".join(sorted(node)) if isinstance(node, dict) else "-"
            raise ValueError(f"--set {key}: unknown setting '{parts[-1]}' (this section has: {known})")
        node[parts[-1]] = val


def load_settings(path: str | Path | None = None, overrides=None) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()) or {}
    apply_overrides(raw, overrides or ())
    tag = raw.get("tag", {})
    cam = raw.get("camera", {})
    est = raw.get("estimate", {})
    return Settings(
        tag_size_m=float(tag.get("size_m", 0.12)),
        mount=CameraMount(**cam.get("mount", {})),
        gate=_gate(raw.get("gate", {}) or {}),
        timing=Timing(**raw.get("timing", {})),
        flight=Flight(**raw.get("flight", {})),
        inlier_radius_m=float(est.get("inlier_radius_m", 0.20)),
        min_obs=int(est.get("min_obs", 3)),
        camera_fps=int(cam.get("fps", 0)),
        camera_quality=int(cam.get("quality", 0)),
        search=dict(raw.get("search", {}) or {}),
        camera_sensor={k: v for k, v in (cam.get("sensor") or {}).items() if v is not None},
    )


def mount_from_mapper(path: str | Path) -> CameraMount:
    """Camera mount from a mapper config (camera.extrinsic_T_B_C), e.g. mapper_config_29_Oct.yaml.

    The live geometry supports a camera that is only pitched nose-down (image right = drone
    right), so the rotation must have that form; anything else is refused rather than guessed."""
    import math
    raw = yaml.safe_load(Path(path).read_text()) or {}
    ext = raw["camera"]["extrinsic_T_B_C"]
    t = [float(v) for v in ext["translation_m"]]
    R = [[float(v) for v in row] for row in ext["rotation_matrix"]]
    pitch = math.atan2(R[2][2], R[0][2])                 # lens axis below the nose direction
    s, c = math.sin(pitch), math.cos(pitch)
    want = [[0.0, -s, c], [1.0, 0.0, 0.0], [0.0, c, s]]
    if any(abs(R[i][j] - want[i][j]) > 1e-3 for i in range(3) for j in range(3)):
        raise ValueError(f"{path}: rotation_matrix is not a pure nose-down pitch with image-right = "
                         f"drone-right; the live pipeline cannot use it")
    return CameraMount(pitch_deg=math.degrees(pitch), offset_fwd_m=t[0], offset_right_m=t[1],
                       offset_down_m=t[2])
