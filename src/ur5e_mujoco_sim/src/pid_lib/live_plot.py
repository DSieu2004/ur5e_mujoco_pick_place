"""One PID dashboard: select a joint to inspect feedback, error and torque."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .plotting import metric_text
from .response import joint_names, numeric, step_metrics


def show_live_response(directory, window_s=8.0):
    import matplotlib
    backends = {name.lower() for name in matplotlib.rcsetup.interactive_bk}
    if matplotlib.get_backend().lower() not in backends:
        raise RuntimeError("PID live plots need an interactive Matplotlib backend such as TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    directory = Path(directory)
    with (directory / "controller_response.csv").open(encoding="utf-8") as stream:
        header = next(csv.reader([stream.readline()]))
        names = joint_names(header)
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.subplots_adjust(left=.34, right=.96, top=.88, bottom=.08, hspace=.22)
        fig.canvas.manager.set_window_title("UR5e PID response")
        title = fig.suptitle("UR5e PID response — waiting for torque samples")
        detail = fig.text(.025, .12, "Select a joint to inspect its PID response.", fontsize=9)
        selector = RadioButtons(fig.add_axes([.015, .40, .23, .27]),
                                [name.removesuffix("_joint") for name in names])
        reference_line, = axes[0].plot([], [], label="Setpoint", color="#df8a22")
        actual_line, = axes[0].plot([], [], label="Measured", color="#1878a5")
        error_line, = axes[1].plot([], [], color="#b84848")
        effort_line, = axes[2].plot([], [], color="#7752a2")
        axes[0].set_ylabel("Angle (deg)")
        axes[1].set_ylabel("Error (deg)")
        axes[2].set_ylabel("u (N·m)")
        axes[2].set_xlabel("Simulation time since first sample (s)")
        axes[0].legend(loc="best")
        for axis in axes:
            axis.grid(alpha=.25)
        for axis in axes[1:]:
            axis.axhline(0, color="grey", linewidth=.6)
        selected = [names[0]]
        time_s, stages = [], []
        data = {name: {key: [] for key in ("reference", "actual", "effort")} for name in names}
        origin = None
        changed = [True]
        was_complete = False

        def select(label):
            selected[0] = f"{label}_joint"
            changed[0] = True
        selector.on_clicked(select)
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            count = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                # Retry a partly written row on the next refresh.
                if not line.endswith("\n"):
                    stream.seek(offset)
                    break
                values = next(csv.reader([line]))
                if len(values) != len(header):
                    continue
                row = dict(zip(header, values))
                stamp = float(row["sim_time_s"])
                if origin is None:
                    origin = stamp
                time_s.append(stamp - origin)
                stages.append(row.get("stage", ""))
                for name in names:
                    for key, suffix in (("reference", "reference_rad"), ("actual", "actual_rad"),
                                        ("effort", "control_effort_nm")):
                        data[name][key].append(numeric(row, f"{name}_{suffix}"))
                count += 1
            complete = (directory / "complete.flag").exists()
            ready = time_s and all(np.any(np.isfinite(data[name]["effort"])) for name in names)
            if complete and not ready:
                plt.close(fig)
                raise ValueError(
                    "Recording has no measured PID torque; use an effort-mode PID run")
            if ready and (count or changed[0] or complete != was_complete):
                name = selected[0]
                # Keep a short time scale during control so transients remain visible.
                lower = 0 if complete else max(0, time_s[-1] - window_s)
                start = int(np.searchsorted(time_s, lower))
                t = np.asarray(time_s[start:])
                series = {key: np.asarray(value[start:]) for key, value in data[name].items()}
                baseline = 0.0
                step_path = directory / "pid_step_test.json"
                definition = None
                if complete and step_path.exists():
                    metadata = json.loads(step_path.read_text())
                    definition = next(
                        (step for step in metadata["steps"] if step["joint"] == name), None)
                    if definition is not None:
                        absolute = t + origin
                        mask = ((absolute >= definition["start_sim_time_s"] - .2)
                                & (absolute < definition["end_sim_time_s"]))
                        series = {key: value[mask] for key, value in series.items()}
                        baseline = definition["initial_reference_rad"]
                        t, metrics = step_metrics(absolute[mask], series["reference"],
                                                  series["actual"], series["effort"], baseline,
                                                  definition["target_rad"])
                        gains = metadata.get("gains", {}).get(name, {})
                        label = "\n".join(f"{key.upper()} = {gains[key]:g}"
                                          for key in ("p", "i", "d") if key in gains)
                        detail.set_text(f"{label}\n\n{metric_text(metrics)}")
                        lower = float(t[0])
                axes[0].set_ylabel("Angle change (deg)" if definition else "Angle (deg)")
                axes[2].set_xlabel("Time since measured step (s)" if definition else
                                   "Simulation time since first sample (s)")
                reference_line.set_data(t, np.degrees(series["reference"] - baseline))
                reference_line.set_drawstyle("steps-post" if definition else "default")
                actual_line.set_data(t, np.degrees(series["actual"] - baseline))
                error_line.set_data(t, np.degrees(series["reference"] - series["actual"]))
                effort_line.set_data(t, series["effort"])
                for axis in axes:
                    axis.relim()
                    axis.autoscale_view()
                    axis.set_xlim(lower, max(lower + 1, float(t[-1])))
                subtitle = "Measured step response" if definition else stages[-1]
                status = 'complete' if complete else 'LIVE'
                title.set_text(f"PID response: {name} — {status}\n{subtitle}")
                fig.canvas.draw_idle()
                changed[0] = False
                was_complete = complete
            plt.pause(.15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--window", type=float, default=8.0, help="live time span in seconds")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    try:
        show_live_response(args.run_directory, args.window)
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
