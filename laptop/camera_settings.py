"""Read and change the drone camera's sensor settings (exposure, gain, brightness, contrast ...).

Needs the camera-settings firmware patch (FLASH_CAMERA_SETTINGS.md) - older firmware never replies.
Works while run_baseline.py / tag_stream.py are running (it uses its own UDP port), so you can
watch the --show window and tune live.

  python camera_settings.py --esp-ip 172.20.10.5                      show the current settings
  python camera_settings.py --esp-ip 172.20.10.5 ae_level=-2 brightness=0
  python camera_settings.py --esp-ip 172.20.10.5 --exposure 300       manual exposure (auto off)
  python camera_settings.py --esp-ip 172.20.10.5 --gain 8             manual gain (auto gain off)
  python camera_settings.py --esp-ip 172.20.10.5 --auto               back to auto exposure + gain
  python camera_settings.py --esp-ip 172.20.10.5 -i                   interactive: type changes

Against glare (bright patches washing out the tags), try in this order:
  1. brightness=0          the firmware boots with +1
  2. ae_level=-2           auto exposure aims darker (-5..5)
  3. --exposure N          fixed exposure: start at about half the "live exposure" shown, then
                           adjust; add --gain to stop auto gain brightening it back up
Settings are lost when the drone reboots; put the ones you like in miniproject/config.yaml
(camera: sensor:) and run_baseline.py applies them at the start of every run.
"""
from __future__ import annotations

import argparse
import csv
import socket
import struct
import sys
import time
from pathlib import Path

CMD_PORT = 5006
PKT_CMD = 0x02
CMD_CAMERA_SETTINGS = 0x0A          # must match main/camera_settings.h
OP_READ, OP_CHANGE = 0, 1
REPLY_MAGIC = b"ECSS"
_HDR = struct.Struct("<4sBBHiiiBB")   # 22 bytes, matches reply_hdr_t
_ITEM = struct.Struct("<Bh")
_RESULT = struct.Struct("<Bb")

# name: (id, low, high, help)  - ranges are the OV3660's
PARAMS = {
    "brightness":  (1, -3, 3, "picture brightness (firmware boots with +1)"),
    "contrast":    (2, -3, 3, "contrast"),
    "sharpness":   (3, -3, 3, "sharpening"),
    "denoise":     (4, 0, 8, "noise reduction"),
    "ae_level":    (5, -5, 5, "auto-exposure target: lower = darker picture"),
    "aec":         (6, 0, 1, "1 = auto exposure, 0 = manual (aec_value)"),
    "aec_value":   (7, 0, 32767, "manual exposure time, sensor lines (used when aec=0)"),
    "agc":         (8, 0, 1, "1 = auto gain, 0 = manual (agc_gain)"),
    "agc_gain":    (9, 0, 64, "manual gain (used when agc=0)"),
    "gainceiling": (10, 0, 1023, "highest gain auto gain may use (raw sensor value)"),
    "aec2":        (11, 0, 1, "night mode / DSP auto exposure"),
    "lenc":        (12, 0, 1, "lens shading correction"),
    "bpc":         (13, 0, 1, "black pixel correction"),
    "wpc":         (14, 0, 1, "white pixel correction"),
    "raw_gma":     (15, 0, 1, "gamma"),
    "vflip":       (16, None, None, "image flipped vertically (read only)"),
    "hmirror":     (17, None, None, "image mirrored (read only)"),
}
NAME_OF = {v[0]: k for k, v in PARAMS.items()}
STATUS = {0: "ok", -1: "REFUSED by the sensor", -2: "unknown / read only"}


class CameraSettingsError(RuntimeError):
    pass


def parse_assignments(items) -> list[tuple[str, int]]:
    """['ae_level=-2', 'brightness=0'] -> [('ae_level', -2), ('brightness', 0)], range-checked."""
    out = []
    for it in items or ():
        if "=" not in it:
            raise CameraSettingsError(f"{it!r}: write it as name=value, e.g. ae_level=-2")
        name, val = (s.strip() for s in it.split("=", 1))
        name = name.lower()
        if name not in PARAMS:
            raise CameraSettingsError(f"unknown camera setting {name!r}; known: {', '.join(PARAMS)}")
        pid, lo, hi, _ = PARAMS[name]
        if lo is None:
            raise CameraSettingsError(f"{name} is read only (the tag maths depends on it)")
        try:
            v = int(float(val))
        except ValueError:
            raise CameraSettingsError(f"{name}={val!r}: needs a whole number") from None
        if not lo <= v <= hi:
            raise CameraSettingsError(f"{name}={v} is outside {lo}..{hi}")
        out.append((name, v))
    return out


def ordered(changes: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Mode switches first, so 'aec_value' after 'aec=0' is not overridden by auto exposure."""
    first = [c for c in changes if c[0] in ("aec", "agc")]
    return first + [c for c in changes if c[0] not in ("aec", "agc")]


def build_request(changes: list[tuple[str, int]]) -> bytes:
    op = OP_CHANGE if changes else OP_READ
    body = b"".join(_ITEM.pack(PARAMS[n][0], v) for n, v in changes)
    return bytes([PKT_CMD, CMD_CAMERA_SETTINGS, op, len(changes)]) + body


def parse_reply(data: bytes) -> dict:
    if len(data) < _HDR.size or data[:4] != REPLY_MAGIC:
        raise CameraSettingsError("not a camera-settings reply")
    magic, ver, drone_id, pid, live_exp, live_gain, exp_max, n_par, n_res = _HDR.unpack_from(data)
    off = _HDR.size
    params, results = {}, []
    for _ in range(n_par):
        i, v = _ITEM.unpack_from(data, off)
        off += _ITEM.size
        params[NAME_OF.get(i, f"id{i}")] = v
    for _ in range(n_res):
        i, st = _RESULT.unpack_from(data, off)
        off += _RESULT.size
        results.append((NAME_OF.get(i, f"id{i}"), st))
    return dict(version=ver, drone_id=drone_id, sensor=f"0x{pid:04X}", params=params,
                live_exposure=None if live_exp < 0 else live_exp,
                live_gain=None if live_gain < 0 else live_gain / 16.0,
                exposure_max=None if exp_max < 0 else exp_max, results=results)


# At most 5 changes per datagram (19 bytes).  Firmware WITHOUT this command passes any unknown
# command of 22+ bytes to its generic handler, which would read the payload as a known-tag list;
# shorter datagrams are simply ignored there.
MAX_PER_PACKET = 5


def request(esp_ip: str, changes=(), port: int = CMD_PORT, timeout: float = 0.8, tries: int = 3) -> dict:
    """Send changes (empty = just read) and return the drone's reply as a dict."""
    changes = ordered(list(changes))
    chunks = [changes[i:i + MAX_PER_PACKET] for i in range(0, len(changes), MAX_PER_PACKET)] or [[]]
    results, reply = [], None
    for chunk in chunks:
        reply = _request_once(esp_ip, chunk, port, timeout, tries)
        results += reply["results"]
    reply["results"] = results
    return reply


def _request_once(esp_ip: str, changes, port: int, timeout: float, tries: int) -> dict:
    pkt = build_request(changes)
    assert len(pkt) < 22
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("0.0.0.0", 0))
        s.settimeout(timeout)
        for _ in range(tries):
            s.sendto(pkt, (esp_ip, port))
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                try:
                    data, addr = s.recvfrom(2048)
                except socket.timeout:
                    break
                except ConnectionResetError:      # Windows: ICMP port unreachable
                    break
                if data[:4] == REPLY_MAGIC:
                    return parse_reply(data)
    raise CameraSettingsError(
        f"no reply from {esp_ip}:{port}. Either the drone is off / not on this network, or its "
        f"firmware does not have the camera-settings command yet (see FLASH_CAMERA_SETTINGS.md)")


def format_reply(r: dict) -> str:
    lines = [f"drone {r['drone_id']}, sensor {r['sensor']}"]
    if r["results"]:
        lines.append("changes: " + ", ".join(f"{n} {STATUS.get(st, st)}" for n, st in r["results"]))
    live = []
    if r["live_exposure"] is not None:
        live.append(f"live exposure {r['live_exposure']}" +
                    (f" of max {r['exposure_max']}" if r["exposure_max"] else ""))
    if r["live_gain"] is not None:
        live.append(f"live gain x{r['live_gain']:.1f}")
    if live:
        lines.append("  ".join(live) + "   (what the sensor is using right now)")
    p = r["params"]
    auto_e = "auto" if p.get("aec") else f"MANUAL ({p.get('aec_value')})"
    auto_g = "auto" if p.get("agc") else f"MANUAL ({p.get('agc_gain')})"
    lines.append(f"exposure {auto_e}, gain {auto_g}, ae_level {p.get('ae_level')}, "
                 f"brightness {p.get('brightness')}, contrast {p.get('contrast')}")
    changed = {n for n, _ in r["results"]}
    if "aec_value" in changed and p.get("aec"):
        lines.append("note: auto exposure is ON, so aec_value is overridden - use --exposure N (sets aec=0)")
    if "agc_gain" in changed and p.get("agc"):
        lines.append("note: auto gain is ON, so agc_gain is overridden - use --gain N (sets agc=0)")
    lines.append("all: " + "  ".join(f"{k}={v}" for k, v in p.items()))
    return "\n".join(lines)


def log_change(esp_ip: str, changes, reply: dict, path: Path) -> None:
    try:
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "esp_ip", "changes", "results", "live_exposure", "live_gain", "settings"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), esp_ip,
                        " ".join(f"{n}={v}" for n, v in changes),
                        " ".join(f"{n}:{STATUS.get(s, s)}" for n, s in reply["results"]),
                        reply["live_exposure"], reply["live_gain"],
                        " ".join(f"{k}={v}" for k, v in reply["params"].items())])
    except OSError:
        pass


def shortcuts(args) -> list[str]:
    extra = []
    if args.auto:
        extra += ["aec=1", "agc=1"]
    if args.exposure is not None:
        extra += ["aec=0", f"aec_value={args.exposure}"]
    if args.gain is not None:
        extra += ["agc=0", f"agc_gain={args.gain}"]
    return extra


def interactive(esp_ip: str, port: int, logpath: Path) -> None:
    print("type name=value (several on one line ok), 'exposure 300', 'gain 8', 'auto', "
          "'show', 'help' or 'q'")
    while True:
        try:
            line = input("camera> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("q", "quit", "exit"):
            return
        if line in ("h", "help", "?"):
            for k, (i, lo, hi, h) in PARAMS.items():
                rng = "read only" if lo is None else f"{lo}..{hi}"
                print(f"  {k:12s} {rng:12s} {h}")
            continue
        words = line.split()
        items = []
        if not words or words[0] == "show":
            items = []
        elif words[0] == "auto":
            items = ["aec=1", "agc=1"]
        elif words[0] in ("exposure", "gain") and len(words) == 2:
            items = (["aec=0", f"aec_value={words[1]}"] if words[0] == "exposure"
                     else ["agc=0", f"agc_gain={words[1]}"])
        else:
            items = words
        try:
            changes = parse_assignments(items)
            r = request(esp_ip, changes, port)
        except CameraSettingsError as e:
            print("  " + str(e))
            continue
        if changes:
            log_change(esp_ip, changes, r, logpath)
        print(format_reply(r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esp-ip", required=True)
    ap.add_argument("changes", nargs="*", help="name=value, e.g. ae_level=-2 brightness=0")
    ap.add_argument("--exposure", type=int, help="manual exposure (turns auto exposure off)")
    ap.add_argument("--gain", type=int, help="manual gain 0-64 (turns auto gain off)")
    ap.add_argument("--auto", action="store_true", help="auto exposure and auto gain back on")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--port", type=int, default=CMD_PORT)
    ap.add_argument("--log", default=str(Path(__file__).resolve().parent / "logs" / "camera_settings_log.csv"))
    a = ap.parse_args()
    if a.interactive:
        interactive(a.esp_ip, a.port, Path(a.log))
        return 0
    try:
        changes = parse_assignments(shortcuts(a) + a.changes)
        r = request(a.esp_ip, changes, a.port)
    except CameraSettingsError as e:
        print(e)
        return 1
    if changes:
        log_change(a.esp_ip, changes, r, Path(a.log))
    print(format_reply(r))
    return 0 if all(st == 0 for _, st in r["results"]) else 2


if __name__ == "__main__":
    sys.exit(main())
