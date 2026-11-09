"""Live camera window (--show) and/or video recording (--record-video) with tag overlays.

Runs in its own thread so drawing, the window and the video file can never slow down or stop
tag detection; the camera thread only hands over the decoded frame and its observations.

Overlay per detected tag (corner outline + label):
    green   CONFIRMED        accepted in >= min_obs distinct frames
    yellow  id N k/3         accepted, still a candidate
    cyan    id 20 HOME n     the home tag, read cleanly (n frames so far) - it is never "confirmed"
                             because it is not a target; it only resets (0, 0) and guides the landing
    red     id N  <reason>   rejected (margin, hamming, too_far, near_edge, ...)
Top bar: time since take-off, frame number, camera fps, drone position from the start mark,
confirmed IDs and candidates.

Video: <run>/camera.mp4 at a fixed frame rate that follows the real clock - when the Wi-Fi
drops frames, the last frame is held, so the video time matches the log's T+ times.

Window keys:  q = close the window only (recording continues; the flight is not affected)
              l = LAND (exactly the same as Ctrl+C in the terminal)
"""
from __future__ import annotations

import _thread
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SCALE = 2                       # 320x240 camera -> 640x480 picture
BAR_H = 44                      # status bar above the image
GREEN, YELLOW, RED, WHITE = (80, 220, 80), (0, 215, 255), (80, 80, 255), (255, 255, 255)
CYAN = (255, 220, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_HOLD_S = 10.0               # longest gap filled with the held frame in the video


def _text(img, s, org, colour=WHITE, scale=0.45, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


class VideoView(threading.Thread):
    def __init__(self, out_dir: Path, show: bool, record: bool, fps: float, min_frames: int,
                 rec=None, header_fn=None, window: str = "drone camera (q = close window, l = LAND)"):
        super().__init__(name="video-view", daemon=True)
        self.out_dir = Path(out_dir)
        self.show, self.record = show, record
        self.fps = max(1.0, float(fps))
        self.min_frames = min_frames
        self.rec = rec
        self.header_fn = header_fn          # () -> str, e.g. drone position; called in submit()
        self.window = window
        self._q: queue.Queue = queue.Queue(maxsize=120)
        self._stop_evt = threading.Event()
        self.ready = threading.Event()
        self.written = 0
        self.received = 0
        self.dropped = 0
        self.path: Path | None = None
        self.notes: list[str] = []
        self._writer_failed = False
        self.cam_fps = 0.0

    # ---- called from the camera thread (must be cheap and never raise) ------------------------
    def submit(self, gray, frame_id: int, t_cap: float, obs: list, tag_map) -> None:
        try:
            dets = []
            for o in obs:
                if o.accepted and getattr(o, "special_seen", None) is not None:
                    n, state = int(o.special_seen), "home"
                elif o.accepted:
                    n = int(o.accepted_frame_count or 0)
                    state = "confirmed" if (tag_map.is_confirmed(o.tag_id) or n >= self.min_frames) else "candidate"
                else:
                    n, state = 0, "rejected"
                dets.append((int(o.tag_id), state, n, o.reason or "", o.range_m,
                             getattr(o, "corners", None), (o.cx, o.cy)))
            conf = tag_map.confirmed_ids()
            cand = [f"{t}({tag_map.frame_count(t)}/{self.min_frames})" for t in tag_map.ids() if t not in conf]
            head = ""
            if self.header_fn is not None:
                try:
                    head = self.header_fn() or ""
                except Exception:
                    head = ""
            epoch = time.time() - (time.monotonic() - t_cap)
            item = (gray.copy(), frame_id, t_cap, epoch, dets, conf, cand, head)
            self.received += 1
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped += 1
        except Exception:
            self.dropped += 1

    def stop(self) -> None:
        self._stop_evt.set()

    # ---- drawing ------------------------------------------------------------------------
    def draw(self, item) -> np.ndarray:
        gray, frame_id, t_cap, epoch, dets, conf, cand, head = item
        h, w = gray.shape[:2]
        img = cv2.cvtColor(cv2.resize(gray, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_GRAY2BGR)
        for tag_id, state, n, reason, rng, corners, centre in dets:
            colour = (GREEN if state == "confirmed" else YELLOW if state == "candidate"
                      else CYAN if state == "home" else RED)
            if corners:
                pts = np.array([[u * SCALE, v * SCALE] for u, v in corners], np.int32)
                cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
                x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
            else:
                x0, y0 = int(centre[0] * SCALE) - 20, int(centre[1] * SCALE) - 20
            if state == "confirmed":
                label = f"id {tag_id} CONFIRMED"
            elif state == "candidate":
                label = f"id {tag_id} {n}/{self.min_frames}"
            elif state == "home":
                label = f"id {tag_id} HOME ok ({n})"
            else:
                label = f"id {tag_id} {reason}"
            if rng is not None:
                label += f" {rng:.2f}m"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            x = min(max(2, x0), img.shape[1] - tw - 6)
            y = max(th + 6, y0 - 6)
            cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (20, 20, 20), -1)
            cv2.putText(img, label, (x, y), FONT, 0.5, colour, 1, cv2.LINE_AA)
        bar = np.zeros((BAR_H, img.shape[1], 3), np.uint8)
        t = self.rec.since_t0(epoch) if self.rec is not None else None
        tt = f"T+{t:6.1f}s" if t is not None else "T+  ---  "
        _text(bar, f"{tt}  frame {frame_id}  {self.cam_fps:4.1f} fps  {head}", (6, 17))
        _text(bar, f"confirmed {conf or '-'}", (6, 37), GREEN)
        _text(bar, f"candidates {', '.join(cand) or '-'}", (250, 37), YELLOW)
        return np.vstack([bar, img])

    # ---- thread -----------------------------------------------------------------------------
    def _note(self, s: str) -> None:
        self.notes.append(s)
        if self.rec is not None:
            try:
                self.rec.note(s)
            except Exception:
                pass

    def _open_writer(self, size):
        self.path = self.out_dir / "camera.mp4"
        wr = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not wr.isOpened():
            self.path = self.out_dir / "camera.avi"
            wr = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size)
        if not wr.isOpened():
            self._note("video: could not open a video file (OpenCV has no encoder); not recording")
            self.path = None
            return None
        return wr

    def run(self) -> None:
        writer = None
        showing = False
        if self.show:
            try:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window, 320 * SCALE, 240 * SCALE + BAR_H)
                blank = np.zeros((240 * SCALE + BAR_H, 320 * SCALE, 3), np.uint8)
                _text(blank, "waiting for camera frames ...", (20, 60))
                cv2.imshow(self.window, blank)
                cv2.waitKey(1)
                showing = True
            except cv2.error as exc:
                self._note(f"video window not available ({str(exc).strip().splitlines()[-1][:80]}); "
                           f"pip install opencv-python (not -headless). Recording still works.")
        self.ready.set()
        last = None                 # newest drawn picture
        next_t = 0.0                # capture time of the next video slot
        times: list[float] = []
        try:
            while not (self._stop_evt.is_set() and self._q.empty()):
                try:
                    item = self._q.get(timeout=0.05)
                except queue.Empty:
                    item = None
                if item is not None:
                    times.append(item[2])
                    times = [x for x in times if item[2] - x < 3.0]
                    if len(times) > 1 and times[-1] > times[0]:
                        self.cam_fps = (len(times) - 1) / (times[-1] - times[0])
                    pic = self.draw(item)
                    if self.record and writer is None and not self._writer_failed:
                        writer = self._open_writer((pic.shape[1], pic.shape[0]))
                        self._writer_failed = writer is None
                    if writer is not None:
                        # video slots every 1/fps s of CAPTURE time; each slot shows the newest
                        # frame captured before it, so stalls appear as a frozen picture
                        t_cap, dt = item[2], 1.0 / self.fps
                        if last is not None:
                            if t_cap - next_t > MAX_HOLD_S:
                                next_t = t_cap - MAX_HOLD_S
                            while next_t < t_cap:
                                writer.write(last)
                                self.written += 1
                                next_t += dt
                        else:
                            next_t = t_cap
                    last = pic
                    if showing:
                        cv2.imshow(self.window, pic)
                if showing:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("l"), ord("L")):
                        self._note("L pressed in the video window -> same as Ctrl+C (land)")
                        _thread.interrupt_main()
                    elif key in (ord("q"), ord("Q")):
                        cv2.destroyWindow(self.window)
                        showing = False
                        self._note("video window closed (recording/flight continue; Ctrl+C in the terminal to land)")
                    else:
                        try:
                            if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                                showing = False
                                self._note("video window closed (recording/flight continue)")
                        except cv2.error:
                            showing = False
        finally:
            if writer is not None:
                if last is not None:
                    writer.write(last)
                    self.written += 1
                writer.release()
            if showing:
                try:
                    cv2.destroyWindow(self.window)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

    def summary(self) -> str:
        if not self.record:
            return f"video window: {self.received} frames shown"
        if self.path is None:
            return "video: not recorded"
        return (f"video saved: {self.path} ({self.written} video frames at {self.fps:g} fps from "
                f"{self.received} camera frames; {self.dropped} dropped)")
