"""Measured controller data and step-response metrics, without a plant model."""

import csv
import json
from pathlib import Path

import numpy as np


def joint_names(header):
    names = [key.removesuffix("_reference_rad") for key in header
             if key.endswith("_reference_rad")]
    if len(names) != 6:
        raise ValueError("PID response requires reference positions for all six arm joints")
    return names


def numeric(row, key):
    value = row.get(key, "")
    return float(value) if value else np.nan


def read_recording(directory):
    with (Path(directory) / "controller_response.csv").open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = joint_names(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError("Controller recording contains no samples")
    if not all(any(np.isfinite(numeric(row, f"{name}_control_effort_nm")) for row in rows)
               for name in names):
        raise ValueError("PID torque output is missing; record an effort-PID run first")
    return names, rows


def joint_series(rows, name):
    return {key: np.array([numeric(row, column) for row in rows])
            for key, column in (("time", "sim_time_s"),
                                ("reference", f"{name}_reference_rad"),
                                ("actual", f"{name}_actual_rad"),
                                ("effort", f"{name}_control_effort_nm"))}


def step_metrics(time_s, reference, actual, effort, initial, target):
    """
    10–90% rise, overshoot, 2% settling and mean final 0.5 s error.

    Time zero is the first observed new controller reference, not publication
    time. A settling result needs at least 0.25 s of observations inside the
    band through the end of the test. Unreached thresholds remain null.
    """
    amplitude = target - initial
    if abs(amplitude) < 1e-8:
        raise ValueError("A step must have a nonzero amplitude")
    if (not all(np.all(np.isfinite(values)) for values in (time_s, reference, actual, effort))
            or not np.all(np.diff(time_s) > 0)):
        raise ValueError("Step measurements must be finite and have increasing timestamps")
    reached = np.flatnonzero(abs(reference - target) <= abs(amplitude) * 0.02)
    if not len(reached):
        raise ValueError("Controller reference never reached the requested step")
    onset = reached[0]
    if np.any(abs(reference[onset:] - target) > abs(amplitude) * 0.02):
        raise ValueError("Reference changes during the step hold; step metrics would be invalid")
    time_s = time_s - time_s[onset]
    normalized = (actual - initial) / amplitude

    def crossing(level):
        indices = np.flatnonzero((time_s >= 0) & (normalized >= level))
        if not len(indices):
            return None
        index = indices[0]
        if index == 0 or normalized[index - 1] >= level:
            return float(time_s[index])
        fraction = (level - normalized[index - 1]) / (normalized[index] - normalized[index - 1])
        return float(max(0.0, time_s[index - 1] + fraction * (time_s[index] - time_s[index - 1])))

    low, high = crossing(0.1), crossing(0.9)
    post = time_s >= 0
    post_time = time_s[post]
    if len(post_time) < 3:
        raise ValueError("Too few samples after the measured step transition")
    error = reference[post] - actual[post]
    outside = np.flatnonzero(abs(error) > abs(amplitude) * 0.02)
    first_inside = int(outside[-1] + 1) if len(outside) else 0
    settling = None
    if first_inside < len(post_time) and post_time[-1] - post_time[first_inside] >= 0.25:
        settling = float(post_time[first_inside])
    tail = post_time >= post_time[-1] - 0.5
    metrics = {
        "rise_time_10_90_s": None if low is None or high is None else high - low,
        "overshoot_percent": float(max(0.0, normalized[post].max() - 1.0) * 100),
        "settling_time_2_percent_s": settling,
        "steady_state_error_rad": float(error[tail].mean()),
        "peak_commanded_torque_nm": float(np.nanmax(abs(effort[post]))),
        "hold_observed_s": float(post_time[-1]),
        "sample_period_median_s": float(np.median(np.diff(post_time))),
        "samples": int(post.sum()),
    }
    return time_s, metrics


def measured_steps(directory, rows):
    path = Path(directory) / "pid_step_test.json"
    if not path.exists():
        return [], {}
    metadata = json.loads(path.read_text())
    steps = []
    for definition in metadata["steps"]:
        selected = [row for row in rows
                    if definition["start_sim_time_s"] - 0.2 <= float(row["sim_time_s"])
                    < definition["end_sim_time_s"]]
        series = joint_series(selected, definition["joint"])
        if len(series["time"]) < 10:
            raise ValueError(f"Insufficient samples for {definition['joint']}")
        series["time"], metrics = step_metrics(
            series["time"], series["reference"], series["actual"], series["effort"],
            definition["initial_reference_rad"], definition["target_rad"])
        steps.append((definition, series, metrics))
    return steps, metadata
