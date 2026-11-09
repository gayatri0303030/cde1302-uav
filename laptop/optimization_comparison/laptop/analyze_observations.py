#!/usr/bin/env python3
"""Adapt a legacy observations ZIP to the existing ground-tag graph and solver.

This is an offline analysis driver, not a replacement optimisation backend.
All camera factors, ground initialisation, SE(3) updates, robust costs and
Gauss--Newton solves are provided by tag_mapping's existing Python modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from tag_mapping.config import load_config
from tag_mapping.comparison import export_comparison
from tag_mapping.data import dump_json, file_hash, software_hash, write_csv
from tag_mapping.graph import build_graph
from tag_mapping.reporting import evaluate, export_results
from tag_mapping.solver import solve
from tag_mapping.synchronization import interpolate_pose, synchronize


def ns(value):
    return int(Decimal(str(value)) * 1_000_000_000)


def read_csv(archive, name):
    return list(csv.DictReader(io.TextIOWrapper(archive.open('observations/' + name))))


class FolderArchive:
    """Lets a run_baseline flight folder (logs/fly_<time>) stand in for observations.zip."""
    def __init__(self, folder):
        self.folder = Path(folder)
    def _p(self, name):
        return self.folder / name.split('/', 1)[1]
    def open(self, name):
        return open(self._p(name), 'rb')
    def read(self, name):
        return self._p(name).read_bytes()
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def remove_px4_position_resets(rows, t_from, min_step_m=0.10, step_per_s=0.6):
    """PX4's estimator sometimes resets: the reported XY jumps with no real motion (F3, 30 Oct:
    2.4 m at lift-off).  Odometry with that step in it bends the whole graph around the single
    anchor, so the step is taken out of every later sample.

    Each new PX4 position sample (a new px4_pos_boot_ms; the real drone sends one a second) is
    compared with the previous one moved on by PX4's own logged velocity.  What the velocity does
    not explain counts as a reset when it is more than max(min_step_m, step_per_s * dt), dt capped
    at 1 s: 0.10 m at 10 Hz, 0.6 m at 1 Hz.  Checked on every real log to 30 Oct: normal flights
    stay below 0.5 m at 1 Hz; only F3's lift-off jump (2.4 m) is caught.  Only after ORIGIN_SET."""
    def f(r, k):
        try:
            return float(r.get(k) or 0.0)
        except ValueError:
            return 0.0
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]['t_epoch_sample']))
    off = np.zeros(2)
    resets = []
    last = None                      # (clock_s, xy, v) of the last distinct PX4 sample
    for i in order:
        r = rows[i]
        t = float(r['t_epoch_sample'])
        boot = r.get('px4_pos_boot_ms', '')
        p = np.array([f(r, 'px4_x_m'), f(r, 'px4_y_m')])
        v = np.array([f(r, 'px4_vx_mps'), f(r, 'px4_vy_mps')])
        if boot not in ('', None):
            clock = f(r, 'px4_pos_boot_ms') / 1000.0
            new_sample = last is None or clock != last[0]
        else:                        # older logs: a new sample = a new position value
            clock = t
            new_sample = last is None or not np.array_equal(p, last[1])
        if new_sample:
            if last is not None and t >= t_from:
                dt = clock - last[0]
                if 0.0 < dt <= 5.0:
                    resid = (p - last[1]) - 0.5 * (v + last[2]) * dt
                    if np.hypot(*resid) > max(min_step_m, step_per_s * min(dt, 1.0)):
                        off += resid
                        resets.append(dict(t_after_origin_s=round(t - t_from, 2), dx_m=round(float(resid[0]), 3),
                                           dy_m=round(float(resid[1]), 3), dt_s=round(dt, 3)))
            last = (clock, p, v)
        if off.any():
            r['px4_x_m'] = repr(float(p[0] - off[0]))
            r['px4_y_m'] = repr(float(p[1] - off[1]))
    return resets


def load_refs(path):
    """tags_ref.csv: id,north_m,east_m per line (# comments and a header allowed)."""
    refs = {}
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split('#')[0].split(',')]
        if len(parts) >= 3 and parts[0].lstrip('-').isdigit():
            refs[int(parts[0])] = [float(parts[1]), float(parts[2]), 0.0]
    return refs


def plot_before_after(out, rows, anchor):
    """before_after.png: tag positions before/after vs truth, and the XY error per tag."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for r in rows:
        tx, ty = r['ground_truth_x_m'], r['ground_truth_y_m']
        ax.plot(ty, tx, 'k+', ms=14, mew=2)
        lab = f"tag {r['tag_id']}" + (' (anchor)' if r['anchor'] else '')
        ax.annotate(lab, (ty, tx), xytext=(7, 7), textcoords='offset points', fontsize=9)
        if 'before_x_m' in r:
            ax.plot(r['before_y_m'], r['before_x_m'], 'o', mfc='none', mec='C1', ms=9)
            ax.plot(r['after_y_m'], r['after_x_m'], 's', mfc='none', mec='C0', ms=9)
            ax.annotate('', xy=(r['after_y_m'], r['after_x_m']), xytext=(r['before_y_m'], r['before_x_m']),
                        arrowprops=dict(arrowstyle='->', color='0.5', lw=1))
    ax.plot([], [], 'k+', ms=10, mew=2, label='tape truth')
    ax.plot([], [], 'o', mfc='none', mec='C1', label='before optimisation')
    ax.plot([], [], 's', mfc='none', mec='C0', label='after optimisation')
    ax.plot(0, 0, 'r*', ms=12, label='start')
    ax.set_xlabel('right of start (m)'); ax.set_ylabel('ahead of start (m)')
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.set_title('Tag positions before vs after')
    ids = [r['tag_id'] for r in rows if 'before_error_m' in r]
    be = [r['before_error_m'] for r in rows if 'before_error_m' in r]
    af = [r['after_error_m'] for r in rows if 'before_error_m' in r]
    x = np.arange(len(ids))
    bx.bar(x - 0.2, be, 0.4, color='C1', label='before')
    bx.bar(x + 0.2, af, 0.4, color='C0', label='after')
    for i, (b, a_) in enumerate(zip(be, af)):
        bx.text(i - 0.2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=8)
        bx.text(i + 0.2, a_, f'{a_:.2f}', ha='center', va='bottom', fontsize=8)
    anchor_ids = set(anchor) if isinstance(anchor, (list, tuple, set)) else {anchor}
    bx.set_xticks(x, [f'tag {i}' + ('\n(anchor)' if i in anchor_ids else '') for i in ids])
    bx.set_ylabel('horizontal error vs tape truth (m)')
    bx.set_title('Error per tag (anchor error is a fit check, not accuracy)')
    bx.grid(axis='y', alpha=.3); bx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'before_after.png', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor', default='1',
                        help="anchor tag id, or several: 5,20 (every one seen in >= 3 frames becomes a fixed "
                             "XY constraint; two far apart also pin the map's rotation), or 'auto' = the "
                             "farthest truth tag with >= 3 accepted frames (never the run's home tag alone)")
    parser.add_argument('--refs', type=Path,
                        help='tags_ref.csv (id,north_m,east_m); default = the 26 Oct lab layout')
    parser.add_argument('--mapper-config', type=Path, default=None,
                        help='base mapper config (default: the run folder mapper_config_used.yaml, else '
                             'mapper_config_current.yaml, else mapper_config_29_Oct.yaml).  A measured camera mount in it is used as is; '
                             'the provisional sketch mount is replaced by the run\'s config_used.yaml mount')
    parser.add_argument('--anchor-sigma-m', type=float, default=0.01)
    parser.add_argument('--init-from', choices=['live', 'floor-ray'], default='live',
                        help="where the solver starts the tag positions: 'live' = the flight's own "
                             "estimate (median of its accepted sightings; default since 3 Nov), "
                             "'floor-ray' = the old pixel-ray guess (went 2-3 m wrong on D4/D5)")
    parser.add_argument('--keyframe-interval-s', type=float, default=1.0)
    parser.add_argument('--odometry-weight-scale', type=float, default=1.0,
                        help='Multiply default odometry sigmas, for sensitivity checks')
    parser.add_argument('--firm-px4', action='store_true',
                        help='EXPERIMENT: hold PX4 movement firmly (never ignored) so the drawn path cannot '
                             'swing metres away.  Not default: tested on D14 / D8 it did not reliably help')
    parser.add_argument('--outlier-m', type=float, default=0.5,
                        help='drop a sighting whose live position is this far from its tag\'s median '
                             '(a misread of another tag in that frame).  0 = off')
    parser.add_argument('--keep-px4-jumps', action='store_true',
                        help='do NOT remove PX4 position resets (sudden XY jumps with no matching '
                             'velocity, e.g. at lift-off) from the odometry')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    opener = FolderArchive if args.archive.is_dir() else zipfile.ZipFile
    with opener(args.archive) as archive:
        recorded_config = yaml.safe_load(archive.read('observations/config_used.yaml'))
        legacy_events = [json.loads(line) for line in archive.read('observations/events.jsonl').decode().splitlines()]
        origin_event = next(e for e in legacy_events if e['event'] == 'ORIGIN_SET')
        telemetry = read_csv(archive, 'px4_telemetry.csv')
        raw = read_csv(archive, 'raw_detections.csv')
        legacy_observations = read_csv(archive, 'observations.csv')
        legacy_summary = json.loads(archive.read('observations/summary.json'))

    # Rebase both positions AND full attitudes before passing data to the mapper.
    # Its NED-labelled mathematical axes are used as forward/right/down here.
    rz = Rotation.from_euler('z', -origin_event['origin_yaw_rad']).as_matrix()
    origin = np.array([origin_event['origin_x'], origin_event['origin_y'], 0.0])
    if not telemetry or telemetry[0].get('px4_qw', '') == '':
        raise SystemExit('px4_telemetry.csv has no PX4 quaternion/z - needs the 125-byte telemetry '
                         '(updated protocol.py + firmware)')
    telemetry = [r for r in telemetry if r.get('px4_qw', '') != '' and r.get('px4_z_m', '') != '']
    run_id = 'legacy-' + (file_hash(args.archive)[:16] if args.archive.is_file()
                          else file_hash(args.archive / 'raw_detections.csv')[:16])
    # Several UDP batches share a host timestamp while the cached attitude (and
    # occasionally position) changes. Retain the final snapshot for that host
    # instant rather than inventing distinct capture times for those rows.
    duplicates = len(telemetry) - len({r['t_epoch_sample'] for r in telemetry})
    telemetry = list({r['t_epoch_sample']: r for r in telemetry}.values())
    resets = [] if args.keep_px4_jumps else remove_px4_position_resets(telemetry, float(origin_event['t_epoch']))
    dump_json(out / 'px4_resets_removed.json', dict(removed=resets, kept=bool(args.keep_px4_jumps)))
    if resets:
        print('PX4 position jumps taken out of the odometry (estimator resets, not real motion): '
              + '; '.join('%.1f s after ORIGIN_SET: (%+.2f, %+.2f) m' % (r['t_after_origin_s'], r['dx_m'], r['dy_m'])
                          for r in resets) + '  [--keep-px4-jumps to keep them]', flush=True)
    event_stream = []
    for k, row in enumerate(telemetry):
        xyz = rz @ (np.array([float(row[f'px4_{a}_m']) for a in 'xyz']) - origin)
        q = Rotation.from_matrix(rz @ Rotation.from_quat(
            [float(row['px4_q' + a]) for a in 'xyzw']).as_matrix()).as_quat()
        envelope = dict(run_id=run_id, kind='position', segment_id=0,
                        host_receive_ns=ns(row['t_epoch_sample']),
                        fc_time_ns=ns(row['t_epoch_sample']), sequence=k)
        # The sample epoch is a host-derived approximate lookup time, not a
        # verified FC/exposure timestamp. synchronize uses host_receive_ns.
        event_stream.append(dict(envelope, **dict(zip('xyz', map(float, xyz)))))
        event_stream.append(dict(envelope, kind='attitude',
                                 **dict(zip(('qx', 'qy', 'qz', 'qw'), map(float, q)))))

    here = Path(__file__).parent
    # the run's own mount first (run_baseline saves it per flight), so a re-run never mixes mounts
    own = args.archive / 'mapper_config_used.yaml' if args.archive.is_dir() else None
    base = args.mapper_config or next(p for p in (own, here / 'mapper_config_current.yaml',
                                                   here / 'mapper_config_29_Oct.yaml',
                                                   here / 'mapper_config.example.yaml') if p and p.exists())
    c = yaml.safe_load(Path(base).read_text())
    c['config_id'] = 'legacy_observations_approximate_ground_tags'
    c['allow_unverified_measurements'] = True
    c['legacy_replay']['allow_assumed_pose'] = True
    c['frames']['start']['input_positions_already_ground_start_relative'] = True
    ext = c['camera']['extrinsic_T_B_C']
    if ext.get('mount_id') == 'provisional_sketch_mount':
        # no measured mount in the base config: use the one the flight itself recorded
        pitch = np.deg2rad(recorded_config['camera']['mount']['pitch_deg'])
        ext['rotation_matrix'] = [
            [0.0, -float(np.sin(pitch)), float(np.cos(pitch))],
            [1.0, 0.0, 0.0],
            [0.0, float(np.cos(pitch)), float(np.sin(pitch))],
        ]
        mount = recorded_config['camera']['mount']
        ext['translation_m'] = [mount['offset_fwd_m'], mount['offset_right_m'], mount['offset_down_m']]
        ext['rotation_description'] = 'Optical right/down/forward to FRD; recorded nose-down pitch'
        ext['measurement_notes'] = 'From archived config_used.yaml, not independently verified'
    rm = np.array(ext['rotation_matrix'], float)
    print('camera mount (%s, from %s): %.3f m forward, %.3f m right, %.3f m down, lens %.1f deg below the nose'
          % (ext.get('mount_id'), Path(base).name, *ext['translation_m'],
             np.degrees(np.arctan2(rm[2, 2], rm[0, 2]))), flush=True)
    c['tag_geometry']['default_size_m'] = recorded_config['tag']['size_m']
    # The original pose lookup already accepted these rows. The longest logged
    # telemetry reception gap is 1.0283 s; keep it explicitly approximate.
    c['timing']['max_bracket_gap_ms'] = 1100.0
    c['optimization']['keyframes']['minimum_interval_s'] = args.keyframe_interval_s
    if args.firm_px4:
        # 5 Nov (D14): PX4's movement between keyframes was held loosely (0.1 m / 0.05-0.08 rad per 0.1 s)
        # AND robustified, so between tag sightings the solved path swung 10-15 m away.  Hold it firmly
        # and never switch it off: PX4's short-term movement is good to a few cm.
        # (held-out check, D14 tag 12 solved as unknown: 0.10 m off with these, 0.31 m loose, 0.31 m at
        # 0.02 rad / 0.04 m)
        c['uncertainty']['fallback_odometry_rotation_sigma_rad'] = [0.05, 0.05, 0.05]
        c['uncertainty']['fallback_odometry_translation_sigma_m'] = [0.1, 0.1, 0.1]
        c['optimization']['robustify_factor_types'] = [
            t for t in c['optimization']['robustify_factor_types'] if t != 'motion']
    for key in ('fallback_odometry_rotation_sigma_rad', 'fallback_odometry_translation_sigma_m'):
        c['uncertainty'][key] = [v * args.odometry_weight_scale for v in c['uncertainty'][key]]
    c['known_tags'] = []
    if args.refs:                     # every truth tag (e.g. a home tag) must pass the ID whitelist
        extra = {int(k) for k in load_refs(args.refs)}
        c['detector']['allowed_tag_ids'] = sorted(set(c['detector']['allowed_tag_ids']) | extra)
    # Search runs: the target IDs are not known in advance - map every ID the flight CONFIRMED
    # (>= 3 good frames; a stray wrong decode seen once or twice would break the solver).
    conf_ids = {int(k) for k, v in (legacy_summary.get('tags') or {}).items()
                if isinstance(v, dict) and v.get('confirmed') and str(k).isdigit()}
    # 5 Nov (strict): ONLY tags the flight confirmed (>= 3 good frames) go into the graph - no default
    # ID list, no unconfirmed one-off reads (D14: tag 20 misread once as 0 was mapped onto tag 20).
    # the home tag and the anchors (refs) are tracked apart from the targets: keep them if seen in >= 3
    # accepted frames
    _n_acc = Counter(int(r['tag_id']) for r in raw if r.get('accepted') == '1')
    _ref_ids = {int(k) for k in load_refs(args.refs)} if args.refs else set()
    conf_ids = set(conf_ids) | {i for i in _ref_ids if _n_acc[i] >= 3}
    c['detector']['allowed_tag_ids'] = sorted(i for i in conf_ids if i < 30) or sorted(c['detector']['allowed_tag_ids'])
    dropped_ids = sorted({int(r['tag_id']) for r in raw if r.get('accepted') == '1'} - set(conf_ids))
    if dropped_ids:
        print(f"left out (not confirmed in flight, misreads): tag(s) {dropped_ids}", flush=True)
    config_path = out / 'config.yaml'
    config_path.write_text(yaml.safe_dump(c, sort_keys=False))
    c = load_config(config_path)
    start_ns = ns(origin_event['t_epoch'])
    positions = [e for e in event_stream if e['kind'] == 'position']
    attitudes = [e for e in event_stream if e['kind'] == 'attitude']
    start_pose = interpolate_pose(positions, attitudes, start_ns, 1_100_000_000, 'host_receive_ns')
    # The marked start XY is from ORIGIN_SET, not interpolated drift.
    start_pose[:2, 3] = 0.0
    # Parked on the floor PX4 z can read 0 or slightly below 0 (height <= 0), which the
    # solver rejects; use at least 2 cm (inside the 5 cm start-height sigma below).
    if -start_pose[2, 3] < 0.02:
        start_pose[2, 3] = -0.02
    start = dict(run_id=run_id, segment_id=0, fc_time_ns=start_ns,
                 host_receive_ns=start_ns, coordinate_convention='NED',
                 native_position_m=start_pose[:3, 3].tolist(),
                 quaternion_xyzw=Rotation.from_matrix(start_pose[:3, :3]).as_quat().tolist(),
                 initial_body_height_above_floor_m=float(-start_pose[2, 3]),
                 initial_body_height_sigma_m=0.05,
                 datum_source='APPROXIMATE: ORIGIN_SET XY; assume native PX4 z=0 is floor; height is telemetry-derived')
    dump_json(out / 'start_event.json', start)
    frames = {}
    accepted_raw = [r for r in raw if r['accepted'] == '1' and int(r['tag_id']) in set(c['detector']['allowed_tag_ids'])]
    # 5 Nov (strict): only PERFECT decodes (hamming 0).  D13-D15: every one-off ghost (0, 13, 16, 21, 26, 27)
    # was a hamming-1 decode; real tags lose only ~3 % of their frames.
    _n = len(accepted_raw)
    accepted_raw = [r for r in accepted_raw if str(r.get('hamming_distance', r.get('hamming', '0'))).strip() in ('0', '0.0', '')]
    if _n - len(accepted_raw):
        print(f"left out {_n - len(accepted_raw)} sighting(s) that needed a 1-bit correction (hamming 1)", flush=True)
    if args.outlier_m > 0:
        # a confirmed ID can still be misread now and then (another tag decoded as it): drop sightings
        # whose live position is far from that tag's median live position
        live = {}
        for r in legacy_observations:
            if r.get('accepted') == '1' and r.get('north_m') not in ('', None):
                live.setdefault(int(r['tag_id']), []).append((r['frame_id'], float(r['north_m']), float(r['east_m'])))
        bad = set()
        for tid, rows in live.items():
            mn = float(np.median([q[1] for q in rows])); me = float(np.median([q[2] for q in rows]))
            bad |= {(f_id, tid) for f_id, n_, e_ in rows if math.hypot(n_ - mn, e_ - me) > args.outlier_m}
        n0 = len(accepted_raw)
        accepted_raw = [r for r in accepted_raw if (r['frame_id'], int(r['tag_id'])) not in bad]
        if n0 - len(accepted_raw):
            print(f"left out {n0 - len(accepted_raw)} sighting(s) more than {args.outlier_m:.1f} m from their "
                  f"tag's other sightings (misreads)", flush=True)
    for row in accepted_raw:
        key = int(row['frame_id'])
        if key not in frames:
            frames[key] = dict(kind='frame', run_id=run_id, segment_id=0,
                               camera_id='camera0', boot_nonce=None, frame_id=key,
                               host_receive_ns=ns(row['camera_capture_timestamp']),
                               capture_time_ns=None, exposure_verified=False,
                               width=320, height=240,
                               calibration_id=c['camera']['calibration_id'],
                               pixel_convention='apriltag_pixel_edges', detections=[])
        frames[key]['detections'].append(dict(
            family_or_dictionary='tag16h5', tag_id=int(row['tag_id']),
            corners=[[float(row[f'corner_{i}_{a}']) for a in 'uv'] for i in range(4)],
            hamming=int(row['hamming_distance']), margin=float(row['decision_margin'])))
    event_stream.extend(frames.values())
    converted = out / 'converted_events.jsonl'
    converted.write_text(''.join(json.dumps(e) + '\n' for e in event_stream))
    prep_end = time.perf_counter()

    # Use the existing synchronizer's explicit legacy path. Never label these
    # timestamps, floor reference, calibration or survey uncertainties verified.
    sync_start = time.perf_counter()
    synchronized, sync_rows, context = synchronize(event_stream, out, c, start_event=start)
    context['warnings'].extend([
        'Axes are start-forward/right/down after rotating native XY and full attitude by minus initial heading.',
        'Legacy host-derived camera and telemetry epochs; exposure timing not verified; telemetry max bracket gap 1.1 s.',
        'Assume native PX4 z=0 is floor; start height is a telemetry proxy, not an independent measurement; sigma=0.05 m.',
        'Calibration is the existing repository 320x240 checkerboard calibration; recording contains no independent calibration identifier.',
        'Ground truth constrains the anchor tag(s) %s XY only; anchor sigma=%.6g m is assumed, not surveyed uncertainty.' % (args.anchor, args.anchor_sigma_m),
        '%d duplicate host-time telemetry snapshots coalesced by retaining the last logged snapshot at each timestamp.' % duplicates,
        'Recording acceptance gates preserved; tags with fewer than three accepted frames are tentative.',
    ])
    sync_end = time.perf_counter()
    initial_start = time.perf_counter()
    initial_graph = build_graph(synchronized, context, c, require_anchor=False)
    initial_end = time.perf_counter()
    before_tags = {k: v.copy() for k, v in initial_graph.initial.tags.items()}

    # Only the selected anchor enters the fit. Load held-out XY truth later.
    truth = load_refs(args.refs) if args.refs else {0: [1.2, .65, 0.], 1: [1.2, -.7, 0.], 2: [2.78, -.7, 0.], 3: [2.7, .52, 0.]}
    seen = Counter(int(r['tag_id']) for r in accepted_raw)
    # The home tag (run_baseline --home-tag) is a check, never the anchor: it sits 0.45 m from the
    # start, and a single anchor that close lets the map swing about it (30 Oct: tags 4-6 m off).
    # Fallback anchor = the FARTHEST truth tag seen in >= 3 frames (far anchors were the steadiest).
    home_id = (legacy_summary.get('home_tag') or {}).get('id')
    home_id = int(home_id) if home_id is not None else None
    candidates = [(float(np.hypot(*truth[t][:2])), seen[t], t) for t in truth
                  if t != home_id and seen[t] >= 3 and ('tag16h5', t) in before_tags]
    usable = lambda t: t in truth and ('tag16h5', t) in before_tags and seen[t] >= 3
    extra_anchors = []
    if str(args.anchor) != 'auto':
        # 5 Nov: every listed anchor that was seen in >= 3 frames is used (e.g. 5,11,12,20)
        asked = [int(v) for v in str(args.anchor).split(',') if v.strip()]
        for t in asked:
            if not usable(t):
                print(f'anchor tag {t} not used: not seen in >= 3 accepted frames (or no tape position)', flush=True)
        ok = [t for t in asked if usable(t)]
        far_ok = [t for t in ok if t != home_id]
        if far_ok:
            args.anchor, extra_anchors = far_ok[0], [t for t in ok if t != far_ok[0]]
        else:
            pick = max(candidates)[2] if candidates else None
            print('none of the map anchors ' + ','.join(str(t) for t in asked if t != home_id)
                  + ' was seen in >= 3 frames: '
                  + (f'using tag {pick} (the farthest tag seen)' if pick is not None else 'no other tag to use'),
                  flush=True)
            args.anchor = 'auto'
            extra_anchors = [t for t in ok if t == home_id]
    if str(args.anchor) == 'auto':
        if not candidates and extra_anchors:
            args.anchor, extra_anchors = extra_anchors[0], extra_anchors[1:]
            print(f'only the home tag {args.anchor} anchors the map this time (it alone cannot fix the '
                  f'rotation well)', flush=True)
        elif not candidates:
            raise SystemExit('no truth tag has >= 3 accepted frames: cannot anchor the optimisation')
        else:
            args.anchor = max(candidates)[2]
    anchor_positions = truth
    anchors = [args.anchor] + [t for t in extra_anchors if t != args.anchor]
    print('anchors (fixed XY constraints): ' + ', '.join(
        f'tag {t} at ({truth[t][0]:+.2f}, {truth[t][1]:+.2f})' for t in anchors), flush=True)
    ca = deepcopy(c)
    ca['known_tags'] = [dict(enabled=True, family_or_dictionary='tag16h5',
                             tag_id=t, frame='ground_start_NED',
                             prior_type='position_xy', position_m=anchor_positions[t],
                             position_sigma_xy_m=[args.anchor_sigma_m] * 2,
                             surveyed_yaw_rad=None, surveyed_yaw_sigma_rad=None,
                             measurement_notes='User-provided picture; uncertainty assumed for this approximate offline calculation')
                        for t in anchors]
    config_path.write_text(yaml.safe_dump(ca, sort_keys=False))
    ca = load_config(config_path)
    anchored_start = time.perf_counter()
    graph = build_graph(synchronized, context, ca, require_anchor=True)
    if args.init_from == 'live':
        # 3 Nov D4/D5: the pixel-ray guess put tags 1-3 m off (rays near the horizon of the 23 deg
        # camera), and the solve then settled in the wrong place.  Start each non-anchor tag at
        # the flight's own estimate instead (it uses the tag's measured distance), in this frame:
        # live positions are measured from the tag-20 re-zeroed start, so add that shift back.
        shift, t_rez = np.zeros(2), float('inf')
        rez = next((e for e in legacy_events if e.get('event') == 'ORIGIN_REZEROED'), None)
        if rez is not None:
            shift = np.array([float(rez.get('px4_offset_north', 0.0)), float(rez.get('px4_offset_east', 0.0))])
            t_rez = float(rez['t_epoch'])
        anchor_keys = {('tag16h5', t) for t in anchors}
        moved = []
        for key in graph.initial.tags:
            # sightings logged after the re-zero are measured from the tag-based zero: shift back
            pts = [np.array([float(r['north_m']), float(r['east_m'])])
                   + (shift if float(r['t_epoch']) >= t_rez else 0.0) for r in legacy_observations
                   if r.get('accepted') == '1' and r.get('north_m') not in ('', None)
                   and int(r['tag_id']) == key[1]]
            if not pts:
                continue
            live_xy = np.median(np.array(pts), axis=0)
            if key in before_tags:                       # "before" = what the flight itself reported
                before_tags[key][:2] = live_xy
            if key not in anchor_keys:
                graph.initial.tags[key][:2] = live_xy
                moved.append(key[1])
        import tag_mapping.comparison as _cmp
        _cmp.BEFORE_NOTE = ("Before = the flight's own (live) estimate: median of its accepted sightings; "
                            "the solver starts the non-anchor tags there.")
        # sightings that cannot be projected from the new start (corner behind the camera) would
        # stop the solve: drop them, as build_graph does for its own start
        from tag_mapping.graph import pixel_factor, tag_size
        kept, dropped = [], 0
        for f in graph.factors:
            if f.kind == 'pixel':
                i, key = f.keys
                try:
                    pixel_factor(graph.initial.poses[i], graph.initial.tags[key], tag_size(graph.config, key),
                                 f.measurement, graph.config, f.metadata['convention'])
                except ValueError:
                    dropped += 1
                    continue
            kept.append(f)
        graph.factors = kept
        print('solver starts tags ' + ', '.join(map(str, sorted(moved))) + " at the flight's own estimate"
              + (f' ({dropped} sightings that do not fit that start left out)' if dropped else ''), flush=True)
    anchored_end = time.perf_counter()
    print(json.dumps(dict(stage='graph_ready', poses=len(graph.initial.poses),
                          tags=len(graph.initial.tags), dimension=graph.dimension,
                          factors=dict(Counter(f.kind for f in graph.factors)),
                          before_tags={str(k): v.tolist() for k, v in before_tags.items()})), flush=True)
    solve_start = time.perf_counter()
    result = solve(graph)
    solve_end = time.perf_counter()
    print(json.dumps(dict(stage='solved', status=result.status,
                          seconds=solve_end-solve_start, conditioning=result.conditioning)), flush=True)

    survey_path = out / 'held_out_survey.yaml'
    survey_path.write_text(yaml.safe_dump(dict(tags=[dict(
        family_or_dictionary='tag16h5', tag_id=k, position_m=p)
        for k, p in truth.items() if k not in anchors])))
    evaluation = evaluate(before_tags, result.state.tags, survey_path, {('tag16h5', t) for t in anchors})
    provenance = dict(archive_sha256=file_hash(args.archive if args.archive.is_file() else args.archive / 'raw_detections.csv'), software_sha256=software_hash(),
                      config_sha256=file_hash(config_path), effective_config=ca,
                      calibration_sha256=file_hash(Path(__file__).parent.parent / 'camera_intrinsics.json'))
    export_results(out, graph, result, context, sync_rows, evaluation, provenance)
    counts = Counter(int(r['tag_id']) for r in accepted_raw)
    rows = []
    for tag_id, gt in truth.items():
        key = ('tag16h5', tag_id)
        row = dict(tag_id=tag_id, anchor=tag_id in anchors,
                   accepted_frames=counts[tag_id], ground_truth_x_m=gt[0], ground_truth_y_m=gt[1])
        if key not in result.state.tags:
            row['status'] = 'unobserved'
        else:
            row['status'] = 'tentative_two_frames' if counts[tag_id] < 3 else result.status
            for stage, state in [('before', before_tags), ('after', result.state.tags)]:
                xy = state[key][:2]
                delta = xy - np.array(gt[:2])
                row.update({stage+'_x_m': float(xy[0]), stage+'_y_m': float(xy[1]),
                            stage+'_error_x_m': float(delta[0]), stage+'_error_y_m': float(delta[1]),
                            stage+'_error_m': float(np.linalg.norm(delta))})
            old = [r for r in legacy_observations if r['accepted'] == '1' and int(r['tag_id']) == tag_id]
            old_mean = np.mean([[float(r['north_m']), float(r['east_m'])] for r in old], axis=0)
            if old:
                row.update(recorded_mean_x_m=float(old_mean[0]), recorded_mean_y_m=float(old_mean[1]),
                           recorded_mean_error_m=float(np.linalg.norm(old_mean-np.array(gt[:2]))))
            live = legacy_summary.get('tags', {}).get(str(tag_id))
            if live:
                row.update(archived_summary_x_m=live['north_m'], archived_summary_y_m=live['east_m'])
        rows.append(row)
    # 5 Nov: the TARGETS too (no tape position, so no error) - where each one was before and after
    for key in sorted(result.state.tags, key=lambda k: k[1]):
        tag_id = key[1]
        if tag_id in truth:
            continue
        row = dict(tag_id=tag_id, anchor=False, accepted_frames=counts[tag_id],
                   ground_truth_x_m=None, ground_truth_y_m=None, status='target_' + result.status)
        for stage, state in [('before', before_tags), ('after', result.state.tags)]:
            if key in state:
                row.update({stage + '_x_m': float(state[key][0]), stage + '_y_m': float(state[key][1])})
        rows.append(row)
    timing = dict(zip_load_and_adapter_s=prep_end-wall_start,
                  synchronization_s=sync_end-sync_start,
                  before_initialization_s=initial_end-initial_start,
                  anchored_graph_build_s=anchored_end-anchored_start,
                  optimization_s=solve_end-solve_start,
                  before_calculation_s=initial_end-initial_start,
                  after_calculation_s=(anchored_end-anchored_start)+(solve_end-solve_start),
                  baseline_pipeline_s=(prep_end-wall_start)+(sync_end-sync_start)+(initial_end-initial_start),
                  optimized_pipeline_excluding_export_s=(prep_end-wall_start)+(sync_end-sync_start)+(anchored_end-anchored_start)+(solve_end-solve_start),
                  driver_through_mapper_export_s=time.perf_counter()-wall_start,
                  clock='time.perf_counter, wall seconds; excludes Python startup/imports; one run')
    for row in rows:
        row['before_batch_calculation_time_s'] = timing['before_calculation_s']
        row['after_batch_calculation_time_s'] = timing['after_calculation_s']
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(out / 'comparison.csv', rows, fields)
    dump_json(out / 'comparison.json', dict(rows=rows, timing=timing, independent_evaluation=evaluation,
               solver_status=result.status, source=args.archive.as_posix(), assumptions=context['warnings'],
               baseline_definition='Existing mapper ray-floor initialization and medoid, before applying any anchor',
               backend='Existing tag_mapping.graph.build_graph and tag_mapping.solver.solve, with rank-check and linear-column-scaling fixes'))
    # The standard exporter seeds the fitting anchor at truth and consequently
    # calls it initial. Keep the genuine pre-anchor values explicitly alongside.
    dump_json(out / 'unanchored_initial_tags.json', {str(k): v.tolist() for k, v in before_tags.items()})
    print(export_comparison(out, rows, timing, result.status), flush=True)
    try:
        plot_before_after(out, [r for r in rows if r.get('ground_truth_x_m') is not None], anchors)
        print(f'Before/after plot: {out / "before_after.png"}', flush=True)
    except Exception as exc:   # the plot is optional; never fail the analysis for it
        print(f'(plot skipped: {exc})', flush=True)
    return 0 if result.status.startswith('converged') else 2


if __name__ == '__main__':
    raise SystemExit(main())
