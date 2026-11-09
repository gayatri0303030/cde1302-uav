"""Passive raw MAVLink + ECAM recorder; sends only camera preview keepalives."""

from __future__ import annotations

import json
from pathlib import Path
import select
import socket
import time

import cv2
import numpy as np

from apriltag_host import Detector, Intrinsics, FIRMWARE_PARAMS, library_path
from camera_stream import FrameAssembler
from protocol import CAMERA_STREAM_PORT, build_camera_stream_command, parse_camera_chunk

from .data import RunWriter, file_hash


def mavlink_event(msg, host_ns, segment):
    envelope = dict(
        segment_id=segment,
        sysid=msg.get_srcSystem(),
        compid=msg.get_srcComponent(),
        sequence=msg.get_seq(),
        host_receive_ns=host_ns,
        status="raw",
        reason="",
    )
    kind = msg.get_type()
    if kind == "LOCAL_POSITION_NED":
        return "position", dict(
            envelope,
            fc_time_ns=int(msg.time_boot_ms) * 1_000_000,
            **{k: getattr(msg, k) for k in ("x", "y", "z", "vx", "vy", "vz")},
        )
    if kind == "ATTITUDE_QUATERNION":
        return "attitude", dict(
            envelope,
            fc_time_ns=int(msg.time_boot_ms) * 1_000_000,
            qw=msg.q1,
            qx=msg.q2,
            qy=msg.q3,
            qz=msg.q4,
            **{k: getattr(msg, k) for k in ("rollspeed", "pitchspeed", "yawspeed")},
        )
    if kind == "ODOMETRY":
        return "odometry", dict(
            envelope,
            fc_time_ns=int(msg.time_usec) * 1000,
            position=[msg.x, msg.y, msg.z],
            quaternion_xyzw=[msg.q[1], msg.q[2], msg.q[3], msg.q[0]],
            frame_id=msg.frame_id,
            child_frame_id=msg.child_frame_id,
            pose_covariance=list(msg.pose_covariance),
            quality=msg.quality,
            reset_counter=msg.reset_counter,
        )
    return None


def json_safe(value):
    """Unknown MAVLink covariances often contain NaN; retain null, never zero."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_jsonl(path):
    if not path:
        return []
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def record(directory, c, esp_ip, duration=0, start_event=None, clock_samples=None):
    from pymavlink.dialects.v20 import common as mavlink

    parts = c["mavlink"]["connection"].split(":")
    if len(parts) != 3 or parts[0] != "udpin":
        raise ValueError("recorder supports udpin:HOST:PORT connections")
    esp_ip = socket.gethostbyname(esp_ip)
    camera, mav, tx = [
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(3)
    ]
    try:
        camera.bind(("0.0.0.0", CAMERA_STREAM_PORT))
        mav.bind((parts[1], int(parts[2])))
        camera.setblocking(False)
        mav.setblocking(False)
        asm = FrameAssembler()
        parsers = {}
        segment = 0
        latest = {}
        reset_counter = None
        k = c["camera"]["K"]
        with RunWriter(directory, c, start_event) as writer, Detector(
            intrinsics=Intrinsics(k[0][0], k[1][1], k[0][2], k[1][2]),
            tag_size_m=c["tag_geometry"]["default_size_m"],
        ) as detector, (Path(directory) / "raw/mavlink.bin").open("xb") as mavraw, (
            Path(directory) / "raw/camera.bin"
        ).open(
            "xb"
        ) as camraw:
            writer.append(
                "detector_metadata",
                family_or_dictionary="tag16h5",
                settings=vars(FIRMWARE_PARAMS),
                library_sha256=file_hash(library_path()),
                pixel_convention="apriltag_pixel_edges",
            )
            for sample in clock_samples or []:
                values = {
                    key: value
                    for key, value in sample.items()
                    if key not in ("kind", "run_id", "schema_version")
                }
                writer.append("clock_samples", **values)
            start_time = time.monotonic()
            keepalive = 0.0
            camera_boot = None
            camera_clock_reset = False
            try:
                while not duration or time.monotonic() - start_time < duration:
                    now = time.monotonic()
                    if now >= keepalive:
                        tx.sendto(build_camera_stream_command(True), (esp_ip, 5006))
                        keepalive = now + 1
                    readable, _, _ = select.select([camera, mav], [], [], 0.1)
                    for sock in readable:
                        data, addr = sock.recvfrom(65535)
                        host_ns = time.monotonic_ns()
                        if sock is camera:
                            if addr[0] != esp_ip:
                                continue
                            offset = camraw.tell()
                            camraw.write(data)
                            writer.append(
                                "camera_datagram",
                                host_receive_ns=host_ns,
                                offset=offset,
                                length=len(data),
                                source=list(addr),
                            )
                            chunk = parse_camera_chunk(data)
                            if chunk is None:
                                writer.append(
                                    "diagnostic",
                                    reason="malformed_or_unsupported_camera_packet",
                                    host_receive_ns=host_ns,
                                )
                                continue
                            if (
                                camera_boot is not None
                                and camera_boot != chunk.boot_nonce
                            ):
                                writer.append(
                                    "diagnostic",
                                    reason="camera_reboot_clock_mapping_invalidated",
                                    host_receive_ns=host_ns,
                                )
                                camera_clock_reset = True
                            camera_boot = chunk.boot_nonce
                            frame = asm.add(chunk, host_ns / 1e9)
                            if frame is None:
                                continue
                            name = f"images/camera_{chunk.boot_nonce}_{frame.frame_id}_{host_ns}.jpg"
                            (writer.path / name).write_bytes(frame.jpeg)
                            image = cv2.imdecode(
                                np.frombuffer(frame.jpeg, np.uint8),
                                cv2.IMREAD_GRAYSCALE,
                            )
                            dets = []
                            reason = ""
                            status = "timing_unverified"
                            if image is None:
                                reason = "jpeg_decode_failed"
                                status = "rejected"
                            else:
                                for d in detector.detect(np.ascontiguousarray(image)):
                                    dets.append(
                                        dict(
                                            family_or_dictionary="tag16h5",
                                            tag_id=d.id,
                                            corners=[list(p) for p in d.corners],
                                            hamming=d.hamming,
                                            margin=d.decision_margin,
                                            status="raw",
                                            reason="",
                                        )
                                    )
                            writer.append(
                                "frame",
                                segment_id=segment,
                                camera_id=c["camera"]["camera_id"],
                                boot_nonce=chunk.boot_nonce,
                                frame_id=frame.frame_id,
                                capture_time_ns=None,
                                clock_domain=c["timing"]["camera_clock_domain"],
                                camera_clock_reset=camera_clock_reset,
                                exposure_verified=False,
                                capture_timestamp_reference=None,
                                host_receive_ns=host_ns,
                                first_receive_ns=round(frame.first_rx * 1e9),
                                width=chunk.width,
                                height=chunk.height,
                                calibration_id=c["camera"]["calibration_id"],
                                image_path=name,
                                age_ms=frame.age_ms,
                                esp_drops=frame.esp_drops,
                                pixel_convention="apriltag_pixel_edges",
                                detections=dets,
                                detector_settings=vars(FIRMWARE_PARAMS),
                                detector_truncated=detector.truncated,
                                status=status,
                                reason=reason,
                            )
                        else:
                            offset = mavraw.tell()
                            mavraw.write(data)
                            writer.append(
                                "mavlink_datagram",
                                host_receive_ns=host_ns,
                                offset=offset,
                                length=len(data),
                                source=list(addr),
                            )
                            if addr not in parsers:
                                parser = mavlink.MAVLink(None)
                                parser.robust_parsing = True
                                parsers[addr] = parser
                            for msg in parsers[addr].parse_buffer(data) or []:
                                if msg.get_type() == "BAD_DATA":
                                    writer.append(
                                        "diagnostic",
                                        reason="malformed_MAVLink",
                                        host_receive_ns=host_ns,
                                    )
                                    continue
                                if (
                                    msg.get_srcSystem() != c["mavlink"]["target_system"]
                                    or msg.get_srcComponent()
                                    != c["mavlink"]["target_component"]
                                ):
                                    continue
                                event = mavlink_event(msg, host_ns, segment)
                                if event is None:
                                    continue
                                kind, values = event
                                stamp = values["fc_time_ns"]
                                reset = (
                                    kind in latest
                                    and stamp < latest[kind] - 1_000_000_000
                                )
                                if kind == "odometry":
                                    reset = reset or (
                                        reset_counter is not None
                                        and values["reset_counter"] != reset_counter
                                    )
                                    reset_counter = values["reset_counter"]
                                if reset:
                                    segment += 1
                                    latest.clear()
                                    writer.append(
                                        "diagnostic",
                                        reason="FC_time_rollback_or_estimator_reset",
                                        segment_id=segment,
                                        host_receive_ns=host_ns,
                                    )
                                latest[kind] = max(stamp, latest.get(kind, stamp))
                                values["segment_id"] = segment
                                writer.append(kind, **json_safe(values))
                    asm.expire(time.monotonic())
            except KeyboardInterrupt:
                pass
            finally:
                tx.sendto(build_camera_stream_command(False), (esp_ip, 5006))
    finally:
        camera.close()
        mav.close()
        tx.close()
    return Path(directory)
