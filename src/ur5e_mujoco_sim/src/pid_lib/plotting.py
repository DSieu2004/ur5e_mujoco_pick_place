"""PID response plots from measured reference, feedback and commanded torque."""

import json
from pathlib import Path

import numpy as np

from .response import joint_series, measured_steps, read_recording


def draw_response(axes, series, initial=0.0, step=False):
    time_s = series["time"]
    reference = np.degrees(series["reference"] - initial)
    actual = np.degrees(series["actual"] - initial)
    axes[0].plot(time_s, reference, label="Setpoint", color="#df8a22", linewidth=1.3,
                 drawstyle="steps-post" if step else "default")
    axes[0].plot(time_s, actual, label="Measured", color="#1878a5", linewidth=1.1)
    axes[0].set_ylabel("Angle change (deg)" if step else "Angle (deg)")
    axes[1].plot(time_s, np.degrees(series["reference"] - series["actual"]), color="#b84848")
    axes[1].set_ylabel("Error (deg)")
    axes[2].plot(time_s, series["effort"], color="#7752a2", linewidth=1)
    axes[2].set_ylabel("u (N·m)")
    axes[2].set_xlabel("Time since measured step (s)" if step else "Simulation time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        if step:
            axis.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].axhline(0, color="grey", linewidth=0.6)
    axes[2].axhline(0, color="grey", linewidth=0.6)


def metric_text(metrics):
    def seconds(key):
        value = metrics[key]
        return "not reached" if value is None else f"{value:.3f} s"
    return (f"Rise 10–90%: {seconds('rise_time_10_90_s')}\n"
            f"Settling 2%: {seconds('settling_time_2_percent_s')}\n"
            f"Overshoot: {metrics['overshoot_percent']:.2f}%\n"
            f"Final error: {np.degrees(metrics['steady_state_error_rad']):.4f}°")


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=150)


def plot_pid_response(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    names, rows = read_recording(directory)
    steps, metadata = measured_steps(directory, rows)
    if steps:
        summary, overview = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        status = "" if metadata.get("completed") else " (incomplete test)"
        summary.suptitle(f"UR5e measured PID step response — {len(steps)} joints{status}",
                         fontsize=16)
        reports = {}
        for axis, (definition, series, metrics) in zip(overview.flat, steps):
            name = definition["joint"]
            initial = definition["initial_reference_rad"]
            target = np.degrees(definition["target_rad"] - initial)
            time_s = series["time"]
            axis.plot(time_s, np.degrees(series["reference"] - initial),
                      color="#df8a22", label="Setpoint", drawstyle="steps-post")
            axis.plot(time_s, np.degrees(series["actual"] - initial),
                      color="#1878a5", label="Measured")
            axis.axhspan(target - abs(target) * .02, target + abs(target) * .02,
                         color="#1878a5", alpha=.10, label="±2% band")
            axis.text(.98, .08, metric_text(metrics), transform=axis.transAxes,
                      ha="right", va="bottom", fontsize=8,
                      bbox={"facecolor": "white", "alpha": .85, "edgecolor": "#dddddd"})
            gains = metadata.get("gains", {}).get(name, {})
            gain_label = "  ".join(f"{key.upper()}={gains[key]:g}"
                                   for key in ("p", "i", "d") if key in gains)
            axis.set_title(f"{name}\n{gain_label}", fontsize=10)
            axis.set_xlabel("Time since measured step (s)")
            axis.set_ylabel("Angle change (deg)")
            axis.grid(alpha=.25)
            axis.legend(loc="upper left", fontsize=8)
            fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
            fig.suptitle(f"PID step response: {name}\n{gain_label}")
            draw_response(axes, series, initial, step=True)
            axes[0].text(.98, .08, metric_text(metrics), transform=axes[0].transAxes,
                         ha="right", va="bottom", fontsize=9)
            save_figure(fig, directory, f"pid_{name}")
            plt.close(fig)
            reports[name] = metrics
        save_figure(summary, directory, "pid_response")
        plt.close(summary)
        (directory / "pid_metrics.json").write_text(json.dumps({
            "source": "measured effort-mode JointTrajectoryController state",
            "test_completed": metadata.get("completed", False), "joints": reports,
            "settling_band_percent": 2, "rise_thresholds_percent": [10, 90],
            "note": ("Metrics have controller-state sampling resolution; "
                     "null means not reached during the hold.")
        }, indent=2) + "\n")
    else:
        fig, axes = plt.subplots(6, 3, figsize=(17, 14), sharex=True, constrained_layout=True)
        fig.suptitle("UR5e PID tracking response — measured feedback, error and torque")
        origin = float(rows[0]["sim_time_s"])
        for index, name in enumerate(names):
            series = joint_series(rows, name)
            series["time"] -= origin
            draw_response(axes[index], series)
            axes[index, 0].set_title(name, fontsize=10)
        save_figure(fig, directory, "pid_response")
        plt.close(fig)
    return directory
