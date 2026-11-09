#!/usr/bin/env python3
"""Record, initialize and optimize floor AprilTags without flight feedback."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from tag_mapping.config import load_config
from tag_mapping.data import (
    dump_json,
    export_csv,
    file_hash,
    load_events,
    software_hash,
    write_csv,
)
from tag_mapping.graph import build_graph
from tag_mapping.recording import read_jsonl
from tag_mapping.reporting import evaluate, export_results
from tag_mapping.solver import Result, solve
from tag_mapping.synchronization import synchronize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument(
        "--operation", choices=["record", "initialize", "optimize"], required=True
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="existing replay run or NEW live run directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new result directory (default: mapping_results/<time_ns>)",
    )
    parser.add_argument("--esp-ip")
    parser.add_argument(
        "--duration", type=float, default=0, help="live seconds; 0 means until Ctrl+C"
    )
    parser.add_argument(
        "--start-event",
        type=Path,
        help="measured start event JSON; replay override leaves raw run unchanged",
    )
    parser.add_argument(
        "--clock-samples",
        type=Path,
        help="measured clock-pair JSONL for live recording",
    )
    parser.add_argument(
        "--frame-metadata",
        type=Path,
        help="verified exposure metadata JSONL for replay",
    )
    args = parser.parse_args(argv)
    output = None
    output_created = False
    try:
        c = load_config(args.config, args.anchors)
        start = json.loads(args.start_event.read_text()) if args.start_event else None
        if args.duration < 0 or not math.isfinite(args.duration):
            raise ValueError("duration must be finite and nonnegative")
        if args.mode == "live":
            if args.operation == "optimize":
                raise ValueError(
                    "optimization is offline: record first, then use --mode replay"
                )
            if not args.esp_ip:
                raise ValueError("live recording requires --esp-ip")
            from tag_mapping.recording import record

            record(
                args.run,
                c,
                args.esp_ip,
                args.duration,
                start,
                read_jsonl(args.clock_samples),
            )
            if args.operation == "record":
                print(f"Raw recordings and CSV: {args.run}")
                return 0
        output = args.output or Path(c["output"]["directory"]) / str(time.time_ns())
        resolved = output.resolve()
        source = args.run.resolve()
        if resolved == source or source in resolved.parents:
            raise ValueError("output must be outside the immutable source run")
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        events = load_events(args.run)
        export_csv(events, output / "csv")
        provenance = dict(
            config_sha256=file_hash(args.config),
            software_sha256=software_hash(),
            events_sha256=file_hash(args.run / "events.jsonl"),
            effective_config=c,
            start_event_sha256=file_hash(
                args.start_event or args.run / c["frames"]["start"]["event_file"]
            ),
        )
        if args.anchors:
            provenance["anchors_sha256"] = file_hash(args.anchors)
        if args.frame_metadata:
            provenance["frame_metadata_sha256"] = file_hash(args.frame_metadata)
        if args.operation == "record":
            dump_json(output / "export.json", provenance)
            print(f"CSV replay export: {output}")
            return 0
        poses, diagnostics, context = synchronize(
            events, args.run, c, read_jsonl(args.frame_metadata), start
        )
        write_csv(
            output / "synchronization.csv",
            diagnostics,
            [
                "frame_id",
                "camera_id",
                "segment_id",
                "status",
                "reason",
                "fc_time_ns",
                "source_time_ns",
                "x",
                "y",
                "z",
                "quaternion_xyzw",
            ],
        )
        graph = build_graph(
            poses, context, c, require_anchor=args.operation == "optimize"
        )
        if args.operation == "optimize" and not c["optimization"]["enabled"]:
            raise ValueError("optimization.enabled is false")
        result = (
            solve(graph)
            if args.operation == "optimize"
            else Result(graph.initial.copy(), "initialized")
        )
        # Evaluation file is deliberately not opened anywhere before fitting finishes.
        survey = Path(c["evaluation"]["held_out_survey_file"])
        if not survey.is_absolute():
            survey = args.config.resolve().parent / survey
        evaluation = (
            evaluate(
                graph.initial.tags,
                result.state.tags,
                survey,
                {f.keys[0] for f in graph.factors if f.kind == "anchor"},
            )
            if args.operation == "optimize"
            else None
        )
        export_results(
            output, graph, result, context, diagnostics, evaluation, provenance
        )
        print(f"{result.status}: {output}")
        return (
            0
            if result.status == "initialized" or result.status.startswith("converged")
            else 2
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        ImportError,
        json.JSONDecodeError,
    ) as exc:
        if output_created and not (output / "failure.json").exists():
            dump_json(output / "failure.json", dict(status="failed", reason=str(exc)))
        print(f"tag_mapper: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
