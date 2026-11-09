"""Export the same before/after measurements to CSV and a terminal table."""

from pathlib import Path

from .data import write_csv



# what the "before" columns are (analyze_observations.py sets it for --init-from live)
BEFORE_NOTE = "Before uses floor-ray initialisation before any anchor is applied."

def export_comparison(directory, rows, timing, solver_status):
    """Times describe a complete batch, never the latency of an individual tag."""
    directory = Path(directory)
    records = []
    for row in rows:
        for stage in ("before", "after"):
            observed = stage + "_x_m" in row
            record = dict(
                role="anchor" if row["anchor"] else ("target" if row["ground_truth_x_m"] is None else "checked"),
                tag_id=row["tag_id"],
                stage=stage,
                anchor=row["anchor"],
                accepted_frames=row["accepted_frames"],
                status=row["status"] if observed else "unobserved",
                ground_truth_x_m=row["ground_truth_x_m"],
                ground_truth_y_m=row["ground_truth_y_m"],
                batch_calculation_time_s=timing[stage + "_calculation_s"],
                batch_pipeline_time_s=timing[
                    "baseline_pipeline_s" if stage == "before"
                    else "optimized_pipeline_excluding_export_s"
                ],
                timing_scope="all_tags_batch_not_per_tag",
            )
            for output, source in (
                ("predicted_x_m", "x_m"),
                ("predicted_y_m", "y_m"),
                ("error_x_m", "error_x_m"),
                ("error_y_m", "error_y_m"),
                ("horizontal_error_m", "error_m"),
            ):
                record[output] = row.get(stage + "_" + source)
            records.append(record)
    fields = [
        "tag_id", "role", "stage", "predicted_x_m", "predicted_y_m",
        "ground_truth_x_m", "ground_truth_y_m", "error_x_m", "error_y_m",
        "horizontal_error_m", "batch_calculation_time_s", "batch_pipeline_time_s",
        "timing_scope", "anchor", "accepted_frames", "status",
    ]
    write_csv(directory / "results.csv", records, fields)
    write_csv(directory / "timing.csv", [
        dict(operation=key, wall_time_s=value)
        for key, value in timing.items() if key.endswith("_s")
    ], ["operation", "wall_time_s"])

    headers = ["Tag", "Stage", "Pred X", "Pred Y", "Truth X", "Truth Y",
               "Error X", "Error Y", "XY error", "Batch s"]
    table = [headers]
    for r in records:
        table.append([str(r["tag_id"]), r["stage"]] + [
            "n/a" if r[key] is None else f"{r[key]:.6f}"
            for key in ("predicted_x_m", "predicted_y_m", "ground_truth_x_m",
                        "ground_truth_y_m", "error_x_m", "error_y_m",
                        "horizontal_error_m", "batch_calculation_time_s")
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = [
        "Before and after optimisation (positions/errors: metres; time: seconds)",
        f"Solver status: {solver_status}",
        "",
    ]
    for i, row in enumerate(table):
        lines.append("  ".join(value.rjust(width) for value, width in zip(row, widths)))
        if i == 0:
            lines.append("  ".join("-" * width for width in widths))
    anchors = ", ".join(str(r["tag_id"]) for r in rows if r["anchor"])
    lines.extend([
        "",
        "Signed errors = prediction - ground truth. XY error = hypot(error X, error Y).",
        "Batch s: before = unanchored initialisation; after = anchored graph build + solve.",
        "Each batch time is repeated for all tags; do not sum it across rows.",
        f"Including shared input preparation/synchronisation: before {timing['baseline_pipeline_s']:.6f} s; "
        f"after {timing['optimized_pipeline_excluding_export_s']:.6f} s.",
        "Python startup/imports and result export are excluded from pipeline times.",
        f"Anchor tag(s): {anchors}. Anchor error is a fit diagnostic, not independent accuracy.",
        BEFORE_NOTE,
        "Approximate replay: exposure timing and floor datum are unverified.",
    ])
    for row in rows:
        if row["status"] in ("unobserved", "tentative_two_frames"):
            lines.append(f"Tag {row['tag_id']}: {row['status']} ({row['accepted_frames']} accepted frames).")
    lines.extend(["", f"CSV results: {directory / 'results.csv'}",
                  f"Timing breakdown: {directory / 'timing.csv'}"])
    summary = "\n".join(lines) + "\n"
    (directory / "terminal_results.txt").write_text(summary, encoding="utf-8")
    return summary
