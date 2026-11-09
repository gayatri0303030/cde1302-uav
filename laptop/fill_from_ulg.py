"""Fill the PX4 columns of raw_detections.csv (z, quaternion, velocity, body rates) from a PX4 .ulg log.

  python fill_from_ulg.py logs/fly_<time> flight.ulg

The telemetry the ESP32 forwards has no attitude quaternion or body rates (and no z or velocity
unless the firmware was updated), so those come from the PX4 log, which has them at 50-250 Hz.

The two clocks are different (laptop wall clock vs PX4 boot time), so the script first finds the
time offset: it lines up the horizontal speed seen in px4_telemetry.csv with the speed in the log's
vehicle_local_position, over every possible offset, and takes the best match. It prints the offset and
how well the x/y tracks agree afterwards. If the match is poor (the drone barely moved, or you picked
the wrong log) it refuses to write the output. Needs: pip install pyulog

Writes raw_detections_ulg.csv next to raw_detections.csv (the original is not touched). For each
detection the PX4 values are the log's NEAREST sample to the camera capture time (no interpolation);
px4_timestamp is that sample's time on the laptop clock.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

UNITS_S = 1e-6


def load_ulg(path: str) -> dict[str, np.ndarray]:
    try:
        from pyulog import ULog
    except ImportError:
        raise SystemExit("pyulog is not installed: pip install pyulog")
    u = ULog(path)
    out = {}
    lp = u.get_dataset("vehicle_local_position").data
    out["lp_t"] = lp["timestamp"] * UNITS_S
    for k in ("x", "y", "z", "vx", "vy", "vz"):
        out["lp_" + k] = lp[k]
    at = u.get_dataset("vehicle_attitude").data
    out["at_t"] = at["timestamp"] * UNITS_S
    for i, k in enumerate(("qw", "qx", "qy", "qz")):
        out["at_" + k] = at[f"q[{i}]"]
    try:
        av = u.get_dataset("vehicle_angular_velocity").data
        out["av_t"] = av["timestamp"] * UNITS_S
        for i, k in enumerate(("roll", "pitch", "yaw")):
            out["av_" + k] = av[f"xyz[{i}]"]
    except (KeyError, IndexError, ValueError):
        out["av_t"] = None
    return out


def speed_series(t, x, y, step=0.05):
    grid = np.arange(t[0], t[-1], step)
    xs, ys = np.interp(grid, t, x), np.interp(grid, t, y)
    sp = np.hypot(np.gradient(xs, step), np.gradient(ys, step))
    k = max(1, int(0.3 / step))
    return grid, np.convolve(sp, np.ones(k) / k, mode="same")


def find_offset(tel_t, tel_x, tel_y, ulg_t, ulg_x, ulg_y, step=0.05):
    """Return (offset, score) with  laptop_epoch = ulg_time + offset."""
    ga, sa = speed_series(tel_t, tel_x, tel_y, step)       # laptop clock
    gb, sb = speed_series(ulg_t, ulg_x, ulg_y, step)       # PX4 clock
    sa, sb = sa - sa.mean(), sb - sb.mean()
    n = len(sa) + len(sb)
    nfft = 1 << (n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(sa, nfft) * np.conj(np.fft.rfft(sb, nfft)), nfft)
    lags = np.concatenate([np.arange(0, len(sa)), np.arange(-len(sb) + 1, 0)])
    idx = np.concatenate([np.arange(0, len(sa)), np.arange(nfft - len(sb) + 1, nfft)])
    best = int(np.argmax(cc[idx]))
    lag = lags[best]
    norm = np.linalg.norm(sa) * np.linalg.norm(sb)
    score = float(cc[idx][best] / norm) if norm > 0 else 0.0
    return ga[0] - gb[0] + lag * step, score


def nearest(t_arr, t):
    i = np.clip(np.searchsorted(t_arr, t), 1, len(t_arr) - 1)
    return np.where(np.abs(t_arr[i - 1] - t) <= np.abs(t_arr[i] - t), i - 1, i)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ulg")
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()
    run = Path(a.run)
    tel = np.genfromtxt(run / "px4_telemetry.csv", delimiter=",", names=True)
    L = load_ulg(a.ulg)
    off, score = find_offset(tel["t_epoch_sample"], tel["px4_x_m"], tel["px4_y_m"],
                             L["lp_t"], L["lp_x"], L["lp_y"])
    print(f"clock offset (laptop epoch - PX4 log time): {off:.3f} s   match score {score:.2f}")
    j = nearest(L["lp_t"] + off, tel["t_epoch_sample"])
    rms = float(np.sqrt(np.mean((L["lp_x"][j] - tel["px4_x_m"]) ** 2 + (L["lp_y"][j] - tel["px4_y_m"]) ** 2)))
    print(f"after alignment, telemetry vs log x/y differ by {rms:.3f} m RMS")
    if score < a.min_score:
        print("match too weak: the drone barely moved, or this is not the log of this run. Nothing written.")
        return 2
    rows = list(csv.DictReader(open(run / "raw_detections.csv")))
    out = run / "raw_detections_ulg.csv"
    for r in rows:
        tc = float(r["camera_capture_timestamp"]) - off          # capture time on the PX4 clock
        i = int(nearest(L["lp_t"], tc))
        r["px4_timestamp"] = f"{L['lp_t'][i] + off:.4f}"
        for k, key in (("px4_x_m", "lp_x"), ("px4_y_m", "lp_y"), ("px4_z_m", "lp_z"),
                       ("px4_vx_mps", "lp_vx"), ("px4_vy_mps", "lp_vy"), ("px4_vz_mps", "lp_vz")):
            r[k] = f"{L[key][i]:.5f}"
        i = int(nearest(L["at_t"], tc))
        for k, key in (("px4_qx", "at_qx"), ("px4_qy", "at_qy"), ("px4_qz", "at_qz"), ("px4_qw", "at_qw")):
            r[k] = f"{L[key][i]:.6f}"
        if L["av_t"] is not None:
            i = int(nearest(L["av_t"], tc))
            for k, key in (("px4_roll_rate_radps", "av_roll"), ("px4_pitch_rate_radps", "av_pitch"),
                           ("px4_yaw_rate_radps", "av_yaw")):
                r[k] = f"{L[key][i]:.6f}"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    return 0


def _selftest() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.1)
    x = np.cumsum(0.2 * np.sin(t / 5) + 0.1 * rng.standard_normal(len(t))) * 0.1
    y = np.cumsum(0.2 * np.cos(t / 3)) * 0.1
    true_off = 1_792_384_000.0 - 523.37            # laptop epoch = ulg time + off
    tu = np.arange(0, 60, 0.02) + 523.37 - 0.0
    xu, yu = np.interp(tu - 523.37, t, x), np.interp(tu - 523.37, t, y)
    off, score = find_offset(t + 1_792_384_000.0, x, y, tu, xu, yu)
    print("selftest offset error", round(off - true_off, 3), "score", round(score, 2))
    assert abs(off - true_off) < 0.1 and score > 0.9


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
