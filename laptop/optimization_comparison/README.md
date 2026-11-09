# Before and after AprilTag optimisation

This bundle contains the Python calculations, the original observations ZIP,
camera calibration, configuration template, and a completed example run.
It runs offline with Python 3.10 or later. No drone connection is needed.

## Run

From this folder after extracting the ZIP:

```bash
python3 -m pip install -r requirements.txt
python3 laptop/run_optimization_comparison.py
```

On Windows, use `python` instead of `python3` if needed.
Each run creates a new `results/run_<timestamp>/` directory and prints a terminal
table. To choose the output directory:

```bash
python3 laptop/run_optimization_comparison.py --output results/my_run
```

The output directory must not already exist. The launcher sets BLAS/OpenMP
threads to one for repeatable timing and uses the bundled recording by default.
An optional `--archive PATH` selects a recording with the same legacy schema.
The adapter uses the supplied tag IDs and ground-truth coordinates below.

## Results

- `results/example_run/results.csv`: one row per tag and stage, with predicted
  X/Y, ground-truth X/Y, signed X/Y errors, horizontal error, batch calculation
  time, batch pipeline time, anchor flag, accepted frame count and status.
- `results/example_run/terminal_results.txt`: the readable terminal table.
- `results/example_run/timing.csv`: detailed wall-clock calculation times.
- `results/example_run/comparison.csv`: one row per tag, with before/after
  values side by side and the original recording's estimates for reference.
- Other files in the example run retain effective configuration, provenance,
  solver diagnostics, iterations, trajectory and map.

All positions and errors are metres. X is forward and Y is right relative to
the recorded start. Signed error = predicted position minus ground truth.
Horizontal error = sqrt(error_x^2 + error_y^2). CSV numbers keep full float
precision; the terminal displays six decimal places. Missing predictions and
errors are blank in CSV and `n/a` in the terminal.

## Meaning of before, after and time

Before = the existing mapper's calibrated corner-ray intersections with the
floor and medoid selection, before applying any anchor. After = the existing
Gauss–Newton joint pose/tag optimisation with tag 1 as an XY anchor. These
compare map estimation before and after fitting, rather than two implementations
of the same calculation. The baseline is not the mean of the archived estimates.

`batch_calculation_time_s` is the wall time to calculate **all tags together**:
before measures the unanchored graph initialisation; after measures anchored
graph construction plus the optimiser, including its rank checks.
Times repeat across rows and must not be summed by tag. They are not per-tag
latencies. `batch_pipeline_time_s` additionally includes shared ZIP loading,
input conversion and synchronisation. The after pipeline excludes the separate
before-only graph build. Python startup/imports and output/report writing are
excluded from these pipeline times. `timing.csv` also records the solve alone
and driver time through the mapper's diagnostic export. Measurements use
`time.perf_counter`, one run on this machine; runtime varies by computer.

The supplied ground-truth tag centres are:

| Tag | X (m) | Y (m) | Role |
| --- | ---: | ---: | --- |
| 0 | 1.20 | 0.65 | Held out for evaluation |
| 1 | 1.20 | -0.70 | Fitting anchor, assumed XY sigma 0.01 m |
| 2 | 2.78 | -0.70 | Held out, tentative: only two accepted frames |
| 3 | 2.70 | 0.52 | Unobserved: no prediction or error available |

The anchor's error is a fit diagnostic and is excluded from independent accuracy
metrics. Ground truth for other tags is used only after fitting. This is an
approximate replay: camera exposure timing, floor datum, mount and survey
uncertainties are not independently verified. The calculation assumes native
PX4 Z=0 is the floor. Further assumptions are saved in `comparison.json`.

## Relevant code and inputs

- `laptop/run_optimization_comparison.py`: portable launcher.
- `laptop/analyze_observations.py`: adapts the recording, times the stages and
  runs the before/after calculation. The supplied truth is defined here.
- `laptop/tag_mapping/`: initialisation, geometry, synchronisation, graph,
  Gauss–Newton solver and result exporters. `comparison.py` owns the CSV and
  terminal table. The existing recorder module is retained for source hashing.
- `laptop/mapper_config.example.yaml`: base calibration and solver settings.
- `camera_intrinsics.json`: calibration provenance.
- `data/observations.zip`: original input archive, copied without modification.
- `laptop/tag_mapper.py`: original mapper entry point, retained because the
  existing provenance hash includes it. Use the offline launcher above for this
  bundle; live recording requires additional repository files and dependencies.
- `tested_environment.json`: Python/platform and imported dependency versions.
- `SHA256SUMS.txt`: hashes of all bundled files except this checksum file.

The solver algorithm is unchanged by the terminal/CSV export additions.
