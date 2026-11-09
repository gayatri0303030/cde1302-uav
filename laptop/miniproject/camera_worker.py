"""Camera thread: ESP32 JPEG stream -> AprilTag detections -> TagPipeline.

This is the same receive path as laptop/tag_stream.py (keepalive, FrameAssembler,
the drone's own detector compiled by apriltag_host.py), without the window.  The
laptop-side stream is used instead of the drone's :5008 detections because it runs on
every frame (~8-9 fps, ~0.13 s old) while the drone's own detector manages ~1.4 Hz and
lags by about a second (see tag_stream.py docstring).
"""

from __future__ import annotations

import select
import socket
import threading
import time

import cv2
import numpy as np

from apriltag_host import Detector, DetectorParams, gate_reason
from camera_stream import COMMAND_PORT, KEEPALIVE_S, FrameAssembler, drain_socket
from protocol import CAMERA_STREAM_PORT, CAMERA_VERSION, build_camera_stream_command

from .recorder import Recorder
from .settings import Settings
from .tag_pipeline import TagPipeline


class CameraWorker(threading.Thread):
    def __init__(self, esp_ip: str, settings: Settings, pipeline: TagPipeline, recorder: Recorder,
                 command_port: int = COMMAND_PORT, camera_port: int = CAMERA_STREAM_PORT, view=None):
        super().__init__(name="camera-worker", daemon=True)
        self.view = view                        # optional VideoView (--show / --record-video)
        self.esp_ip = esp_ip
        self.s = settings
        self.pipeline = pipeline
        self.rec = recorder
        self.command_port = command_port
        self.camera_port = camera_port
        self._stop_evt = threading.Event()
        self.ready = threading.Event()          # set once the socket is bound
        self.error: str | None = None
        # live status
        self.frames = 0
        self.last_frame_t = 0.0
        self.last_age_ms = 0
        self.fps = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:                # surface it instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self.rec.event("CAMERA_ERROR", self.error)
            self.ready.set()

    def _loop(self) -> None:
        cfg = self.s
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        rx.bind(("0.0.0.0", self.camera_port))
        rx.setblocking(False)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = (self.esp_ip, self.command_port)
        keepalive = build_camera_stream_command(True, cfg.camera_fps, cfg.camera_quality)
        asm = FrameAssembler()
        times: list[float] = []
        warned_version = False
        stalled = False
        next_keepalive = 0.0
        det = Detector(DetectorParams(), tag_size_m=cfg.tag_size_m,
                       min_margin=cfg.gate.min_margin, max_hamming=cfg.gate.max_hamming)
        try:                                            # 9 Nov: say which camera numbers are in use
            from apriltag_host import CAMERA_INTRINSICS_SOURCE as _src
            k = det.intrinsics
            self.rec.note(f"camera intrinsics: fx {k.fx:.1f} fy {k.fy:.1f} cx {k.cx:.1f} cy {k.cy:.1f}"
                          f"{', lens distortion corrected' if det.dist else ''} - from {_src}")
        except Exception:
            pass
        self.ready.set()
        try:
            while not self._stop_evt.is_set():
                now = time.monotonic()
                if now >= next_keepalive:
                    next_keepalive = now + KEEPALIVE_S
                    try:
                        tx.sendto(keepalive, target)
                    except OSError:
                        pass                    # drone off / not reachable yet: keep trying
                select.select([rx], [], [], 0.02)
                frame, _, bad_version = drain_socket(rx, asm, self.esp_ip)
                if bad_version is not None and not warned_version:
                    warned_version = True
                    self.rec.event("CAMERA_WARNING", f"drone sends ECAM v{bad_version}, "
                                   f"laptop expects v{CAMERA_VERSION}: reflash the drone")
                if frame is not None:
                    arr = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if arr is None:
                        asm.dropped += 1
                    else:
                        gray = np.ascontiguousarray(arr)
                        dets = det.detect(gray)
                        # capture time = when the frame reached us minus how old it already was
                        t_cap = frame.first_rx - frame.age_ms / 1000.0 + cfg.timing.capture_correction_s
                        obs = []
                        try:
                            obs = self.pipeline.process(frame.frame_id, t_cap, dets, gate_reason)
                        except Exception as exc:    # one bad frame must never stop the camera
                            self.rec.event("FRAME_ERROR", f"frame {frame.frame_id}: {type(exc).__name__}: {exc}")
                        if self.view is not None:   # hands over a copy; never blocks
                            self.view.submit(gray, frame.frame_id, t_cap, obs or [], self.pipeline.tag_map)
                        self.frames += 1
                        self.last_frame_t = time.monotonic()
                        self.last_age_ms = frame.age_ms
                        times.append(self.last_frame_t)
                        times = [t for t in times if self.last_frame_t - t < 3.0]
                        if len(times) > 1 and times[-1] > times[0]:   # Windows clock ticks ~16 ms:
                            self.fps = (len(times) - 1) / (times[-1] - times[0])  # equal stamps after a stall
                        stalled = False
                elif self.last_frame_t and time.monotonic() - self.last_frame_t > 3.0 and not stalled:
                    stalled = True
                    self.rec.event("CAMERA_STALL", "no camera frame for 3 s")
                asm.expire(time.monotonic())
        finally:
            try:
                tx.sendto(build_camera_stream_command(False), target)
            except OSError:
                pass
            det.close()
            rx.close()
            tx.close()
