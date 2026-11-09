"""Immutable JSONL/raw sources and precision-preserving tabular exports."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid

from . import SCHEMA_VERSION


def dump_json(path, value):
    with Path(path).open("x", encoding="utf-8") as h:
        json.dump(value, h, indent=2, allow_nan=False)
        h.write("\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def software_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    h.update((Path(__file__).parents[1] / "tag_mapper.py").read_bytes())
    return h.hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Python repr/CSV uses round-trip float precision; no display rounding.
            writer.writerow(
                {
                    k: (
                        json.dumps(v, allow_nan=False)
                        if isinstance(v, (list, dict))
                        else v
                    )
                    for k, v in row.items()
                }
            )


def load_events(run):
    result = []
    with (Path(run) / "events.jsonl").open(encoding="utf-8") as h:
        for n, line in enumerate(h, 1):
            item = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON number: {value}")
                ),
            )
            if (
                item.get("schema_version") != SCHEMA_VERSION
                or "kind" not in item
                or "run_id" not in item
            ):
                raise ValueError(f"events.jsonl:{n}: invalid event envelope")
            result.append(item)
    if len({e["run_id"] for e in result}) > 1:
        raise ValueError("mixed run IDs")
    return result


COMMON = [
    "run_id",
    "segment_id",
    "sysid",
    "compid",
    "sequence",
    "fc_time_ns",
    "host_receive_ns",
    "status",
    "reason",
]
CSV_SCHEMAS = {
    "position": COMMON + ["x", "y", "z", "vx", "vy", "vz"],
    "attitude": COMMON
    + ["qw", "qx", "qy", "qz", "rollspeed", "pitchspeed", "yawspeed"],
    "odometry": COMMON
    + [
        "frame_id",
        "child_frame_id",
        "position",
        "quaternion_xyzw",
        "pose_covariance",
        "quality",
        "reset_counter",
    ],
    "frames": [
        "run_id",
        "segment_id",
        "camera_id",
        "boot_nonce",
        "frame_id",
        "capture_time_ns",
        "clock_domain",
        "host_receive_ns",
        "width",
        "height",
        "calibration_id",
        "pixel_convention",
        "image_path",
        "age_ms",
        "esp_drops",
        "detection_count",
        "status",
        "reason",
    ],
    "detections": [
        "run_id",
        "segment_id",
        "camera_id",
        "frame_id",
        "family_or_dictionary",
        "tag_id",
        "hamming",
        "margin",
    ]
    + [f"c{i}_{axis}" for i in range(4) for axis in ("u", "v")]
    + ["status", "reason"],
    "clock_samples": [
        "run_id",
        "camera_time_ns",
        "fc_time_ns",
        "sigma_ns",
        "scale",
        "offset_ns",
        "status",
        "reason",
    ],
}


def export_csv(events, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = {key: [] for key in CSV_SCHEMAS}
    for event in events:
        kind = event["kind"]
        if kind in ("position", "attitude", "odometry", "clock_samples"):
            rows[kind].append(event)
        elif kind == "frame":
            rows["frames"].append(
                dict(event, detection_count=len(event.get("detections", [])))
            )
            for d in event.get("detections", []):
                row = {
                    k: event.get(k)
                    for k in ("run_id", "segment_id", "camera_id", "frame_id")
                }
                row.update(d)
                for i, corner in enumerate(d["corners"]):
                    row[f"c{i}_u"], row[f"c{i}_v"] = corner
                rows["detections"].append(row)
    for key, fields in CSV_SCHEMAS.items():
        write_csv(directory / f"{key}.csv", rows[key], fields)


class RunWriter:
    def __init__(self, directory, config, start_event=None):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "images").mkdir()
        (self.path / "raw").mkdir()
        self.run_id = str(uuid.uuid4())
        dump_json(
            self.path / "manifest.json",
            dict(
                schema_version=SCHEMA_VERSION,
                run_id=self.run_id,
                created_unix_ns=time.time_ns(),
                software_hash=software_hash(),
                python=platform.python_version(),
                config_sha256=hashlib.sha256(
                    json.dumps(config, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                dependencies={
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "PyYAML")
                },
                config=config,
                timing_status="unverified_until_exposure_metadata_supplied",
            ),
        )
        start = dict(start_event or {})
        if start.get("run_id") not in (None, self.run_id):
            start["source_run_id"] = start["run_id"]
        start.update(run_id=self.run_id)
        start.setdefault(
            "status", "missing_ground_reference" if not start_event else "supplied"
        )
        dump_json(self.path / "start_event.json", start)
        self.handle = (self.path / "events.jsonl").open("x", encoding="utf-8")

    def append(self, kind, **values):
        event = dict(
            values, kind=kind, run_id=self.run_id, schema_version=SCHEMA_VERSION
        )
        self.handle.write(json.dumps(event, allow_nan=False) + "\n")
        self.handle.flush()
        return event

    def close(self):
        if not self.handle.closed:
            self.handle.close()
            events = load_events(self.path)
            export_csv(events, self.path / "csv")
            hashes = {
                str(p.relative_to(self.path)): file_hash(p)
                for p in sorted(self.path.rglob("*"))
                if p.is_file()
            }
            dump_json(self.path / "hashes.json", hashes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
