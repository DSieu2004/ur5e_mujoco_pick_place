"""Capture time-stamped joint response and controller output commands."""

import csv
import json
import math
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from control_msgs.msg import JointTrajectoryControllerState
from rclpy.executors import ExternalShutdownException, ShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from .timing import describe_trajectory


class TrajectoryRecorder:
    def __init__(self, node, joint_names):
        self.node = node
        self.joint_names = tuple(joint_names)
        self.segments = []
        self.sample_count = 0
        self.pid_sample_count = 0
        self.live_plot_started = False
        self.active = None
        self.lock = threading.Lock()
        self.output_dir = Path.cwd() / "trajectory_runs" / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["sim_time_s", "segment_index", "stage", "segment_elapsed_s"]
        for name in self.joint_names:
            self.fieldnames.extend((f"{name}_reference_rad", f"{name}_actual_rad",
                                    f"{name}_error_rad",
                                    f"{name}_control_effort_nm",
                                    f"{name}_control_position_rad"))
        self.stream = (self.output_dir / "controller_response.csv").open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.node.get_logger().info(
            f"Recording PID feedback and controller commands in {self.output_dir}"
        )
        self.record_node = Node(
            "ur5e_trajectory_recorder",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.subscription = self.record_node.create_subscription(
            JointTrajectoryControllerState,
            "/ur5e_controller/controller_state",
            self.on_controller_state,
            50,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.record_node)
        self.worker = threading.Thread(target=self._spin, daemon=True)
        self.worker.start()

    def _spin(self):
        try:
            self.executor.spin()
        except (ExternalShutdownException, ShutdownException):
            pass

    def _log(self, level, message):
        if self.node.context.ok():
            getattr(self.node.get_logger(), level)(message)
        else:
            print(message, flush=True)

    def _start_live_plot(self):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.node.get_logger().warning(
                "No graphical display detected; saving PID data without a live window"
            )
            return
        # CMake installs PROGRAMS without their .py suffix. Retain the source
        # path as a fallback for direct source execution and symlink installs.
        entry = Path(__file__).absolute().parents[1] / "plot_pid_response_live"
        if not entry.is_file():
            entry = Path(__file__).resolve().parents[1] / "plot_pid_response_live.py"
        try:
            with (self.output_dir / "live_plot.log").open("w", encoding="utf-8") as log:
                subprocess.Popen(
                    [sys.executable, str(entry), str(self.output_dir)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            self.node.get_logger().info("Opening PID feedback/error/torque dashboard")
        except OSError as error:
            self.node.get_logger().warning(f"Live plot could not start: {error}")

    def begin(self, stage, trajectory):
        description = describe_trajectory(trajectory, stage)
        with self.lock:
            if self.active is not None:
                raise RuntimeError("A trajectory is already being recorded")
            description["segment_index"] = len(self.segments)
            description["start_sim_time_s"] = self.node.get_clock().now().nanoseconds * 1e-9
            description["result"] = "running"
            self.segments.append(description)
            self.active = description
        self.node.get_logger().info(
            f"Timed trajectory {stage}: {len(trajectory.points)} waypoints, "
            f"duration {description['duration_s']:.3f} s"
        )

    def end(self, result):
        with self.lock:
            if self.active is not None:
                self.active["result"] = result
                self.active = None

    @staticmethod
    def _positions(point, count):
        positions = getattr(point, "positions", ())
        return list(positions) if len(positions) == count else None

    def on_controller_state(self, message):
        names = list(message.joint_names)
        if not names:
            return
        reference = self._positions(getattr(message, "reference", None), len(names))
        feedback = self._positions(getattr(message, "feedback", None), len(names))
        if reference is None or feedback is None:
            reference = self._positions(getattr(message, "desired", None), len(names))
            feedback = self._positions(getattr(message, "actual", None), len(names))
        if reference is None or feedback is None:
            return
        # Only output contains the command sent by JTC to ros2_control.
        # reference.effort is a trajectory feedforward term, not the PID output.
        output = getattr(message, "output", None)
        effort = getattr(output, "effort", ())
        effort = effort if len(effort) == len(names) else None
        command_position = self._positions(output, len(names))
        stamp = message.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        if stamp_s <= 0:
            stamp_s = self.record_node.get_clock().now().nanoseconds * 1e-9
        name_to_index = {name: index for index, name in enumerate(names)}
        if any(name not in name_to_index for name in self.joint_names):
            return
        with self.lock:
            segment = self.active
            row = {
                "sim_time_s": stamp_s,
                "segment_index": segment["segment_index"] if segment else -1,
                "stage": segment["stage"] if segment else "idle",
                "segment_elapsed_s": max(0.0, stamp_s - segment["start_sim_time_s"])
                if segment else "",
            }
            for name in self.joint_names:
                index = name_to_index[name]
                row[f"{name}_reference_rad"] = reference[index]
                row[f"{name}_actual_rad"] = feedback[index]
                row[f"{name}_error_rad"] = reference[index] - feedback[index]
                for suffix, command in (("control_effort_nm", effort),
                                        ("control_position_rad", command_position)):
                    value = command[index] if command is not None else math.nan
                    row[f"{name}_{suffix}"] = value if math.isfinite(value) else ""
            self.writer.writerow(row)
            self.sample_count += 1
            if effort is not None:
                self.pid_sample_count += 1
                if not self.live_plot_started:
                    self.live_plot_started = True
                    self._start_live_plot()

    def finish(self):
        if self.active is not None:
            self.end("interrupted")
        self.executor.shutdown()
        self.worker.join(timeout=2.0)
        self.record_node.destroy_node()
        self.stream.close()
        (self.output_dir / "planned_trajectories.json").write_text(
            json.dumps(self.segments, indent=2) + "\n", encoding="utf-8"
        )
        (self.output_dir / "complete.flag").touch()
        if not self.sample_count:
            self._log(
                "warning",
                "No /ur5e_controller/controller_state samples were received; "
                f"planned timestamps saved in {self.output_dir}"
            )
            return self.output_dir
        if not self.pid_sample_count:
            self._log(
                "warning",
                "No measured effort-mode PID output; CSV saved without PID plots"
            )
            return self.output_dir
        graphs_created = False
        try:
            from pid_lib.plotting import plot_pid_response

            plot_pid_response(self.output_dir)
            graphs_created = True
        except ImportError as error:
            self._log(
                "warning",
                f"PID data saved, but plotting dependencies are unavailable: {error}"
            )
        self._log(
            "info",
            f"Saved {self.sample_count} controller samples"
            + (" and PID response graphs" if graphs_created else "")
            + f" to {self.output_dir}"
        )
        return self.output_dir
