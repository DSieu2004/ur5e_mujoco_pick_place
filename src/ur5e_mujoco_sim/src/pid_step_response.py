#!/usr/bin/env python3
"""Measure six small joint steps on the running MuJoCo effort-PID controller."""

import json
import math
import time
import traceback

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pick_place_lib.config import HOME_JOINTS
from trajectory_lib.recorder import TrajectoryRecorder
from trajectory_lib.lifecycle import init_task_context


class PIDStepExperiment:
    def __init__(self):
        self.node = Node("ur5e_pid_step_test",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.node.declare_parameter("step_rad", 0.05)
        self.node.declare_parameter("hold_s", 3.0)
        self.amplitude = self.node.get_parameter("step_rad").value
        self.hold_s = self.node.get_parameter("hold_s").value
        if not math.isfinite(self.amplitude) or not 0.01 <= abs(self.amplitude) <= 0.10:
            raise ValueError("step_rad must have magnitude between 0.01 and 0.10 rad")
        if not math.isfinite(self.hold_s) or not 2.0 <= self.hold_s <= 15.0:
            raise ValueError("hold_s must be between 2 and 15 simulation seconds")
        self.latest = None
        self.subscription = self.node.create_subscription(
            JointTrajectoryControllerState, "/ur5e_controller/controller_state", self.receive, 50)
        self.publisher = self.node.create_publisher(
            JointTrajectory, "/ur5e_controller/joint_trajectory", 10)
        self.parameters = self.node.create_client(GetParameters, "/ur5e_controller/get_parameters")
        self.recorder = None
        self.metadata = {"completed": False, "steps": [], "step_rad": self.amplitude,
                         "hold_s": self.hold_s, "controller": "/ur5e_controller",
                         "command_transition_s": 0.01,
                         "input": ("one 10 ms setpoint transition, then constant hold; "
                                   "zero reference velocity")}

    def receive(self, message):
        self.latest = message

    def sim_time(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def hold(self, seconds):
        end = self.sim_time() + seconds
        deadline = time.monotonic() + seconds * 4 + 5
        while rclpy.ok() and self.sim_time() < end:
            if time.monotonic() > deadline:
                raise RuntimeError("Simulation clock stopped or is too slow during the PID test")
            rclpy.spin_once(self.node, timeout_sec=.03)
        if not rclpy.ok():
            raise RuntimeError("ROS stopped during the PID test")

    def save_metadata(self):
        (self.recorder.output_dir / "pid_step_test.json").write_text(
            json.dumps(self.metadata, indent=2) + "\n")

    def prepare(self):
        deadline = time.monotonic() + 25
        while self.latest is None or not self.publisher.get_subscription_count():
            if time.monotonic() > deadline:
                raise RuntimeError("Effort controller is not ready; start a fresh PID simulation")
            rclpy.spin_once(self.node, timeout_sec=.1)
        message = self.latest
        reference = message.reference.positions or message.desired.positions
        feedback = message.feedback.positions or message.actual.positions
        if len(message.output.effort) != len(HOME_JOINTS):
            raise RuntimeError("Step testing requires actual effort-mode PID output")
        indices = {name: index for index, name in enumerate(message.joint_names)}
        for name, target in HOME_JOINTS.items():
            index = indices[name]
            if max(abs(reference[index] - target), abs(feedback[index] - target)) > .02:
                raise RuntimeError(
                    "Start a fresh simulation at home, with pick-and-place disabled")
        if not self.parameters.wait_for_service(timeout_sec=5):
            raise RuntimeError("Controller parameter service is unavailable")
        names = ["command_interfaces"] + [f"gains.{name}.{key}" for name in HOME_JOINTS
                                          for key in ("p", "i", "d")]
        request = GetParameters.Request()
        request.names = names
        future = self.parameters.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5)
        if not future.done() or future.result() is None:
            raise RuntimeError("Could not read the running controller gains")
        values = dict(zip(names, future.result().values))
        if list(values["command_interfaces"].string_array_value) != ["effort"]:
            raise RuntimeError("The active controller does not command effort")
        gains = {}
        for name in HOME_JOINTS:
            gains[name] = {}
            for key in ("p", "i", "d"):
                value = values[f"gains.{name}.{key}"]
                if value.type not in (2, 3):
                    raise RuntimeError(f"Missing runtime gain: {name}.{key}")
                numeric = value.double_value if value.type == 3 else value.integer_value
                gains[name][key] = float(numeric)
        self.metadata["gains"] = gains
        self.recorder = TrajectoryRecorder(self.node, HOME_JOINTS)
        self.save_metadata()

    def command(self, positions, stage, duration_s=.01):
        trajectory = JointTrajectory()
        trajectory.joint_names = list(HOME_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(positions[name]) for name in HOME_JOINTS]
        point.velocities = [0.0] * 6
        point.time_from_start.sec = int(duration_s)
        point.time_from_start.nanosec = int((duration_s % 1) * 1e9)
        trajectory.points = [point]
        self.recorder.begin(stage, trajectory)
        self.publisher.publish(trajectory)

    def run(self):
        self.prepare()
        self.hold(1.0)
        for name, home in HOME_JOINTS.items():
            target = dict(HOME_JOINTS)
            target[name] += self.amplitude
            self.node.get_logger().info(f"PID STEP {name}: {self.amplitude:+.3f} rad")
            start = self.sim_time()
            self.command(target, f"PID step {name}")
            self.hold(self.hold_s)
            self.recorder.end("held")
            self.metadata["steps"].append({"joint": name, "start_sim_time_s": start,
                                           "end_sim_time_s": self.sim_time(),
                                           "initial_reference_rad": home,
                                           "target_rad": target[name]})
            self.save_metadata()
            self.command(HOME_JOINTS, f"PID return {name}")
            self.hold(self.hold_s)
            self.recorder.end("returned")
        feedback = self.latest.feedback.positions or self.latest.actual.positions
        actual = dict(zip(self.latest.joint_names, feedback))
        error = max(abs(actual[name] - value) for name, value in HOME_JOINTS.items())
        self.metadata["final_home_error_rad"] = error
        if error > .002:
            raise RuntimeError(f"Final home error {error:.5f} rad exceeds 0.002 rad")
        self.metadata["completed"] = True
        self.save_metadata()
        self.node.get_logger().info("Six measured PID step responses completed; arm returned home")


def main():
    init_task_context()
    experiment = None
    try:
        experiment = PIDStepExperiment.__new__(PIDStepExperiment)
        experiment.__init__()
        experiment.run()
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        if experiment is not None and hasattr(experiment, "metadata"):
            experiment.metadata["interrupted"] = True
            if experiment.recorder is not None:
                experiment.save_metadata()
        print("PID experiment interrupted; saving partial measurements.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if experiment is not None and getattr(experiment, "recorder", None) is not None:
            if not experiment.metadata["completed"] and rclpy.ok():
                experiment.recorder.end("interrupted")
                if experiment.metadata.get("interrupted") and experiment.latest is not None:
                    feedback = (experiment.latest.feedback.positions
                                or experiment.latest.actual.positions)
                    positions = dict(zip(experiment.latest.joint_names, feedback))
                    experiment.command(positions, "PID interrupted hold", .05)
                else:
                    experiment.command(HOME_JOINTS, "PID interrupted return", .5)
                    try:
                        experiment.hold(1.0)
                    except RuntimeError:
                        pass
                experiment.recorder.end("interrupted")
            experiment.recorder.finish()
        if experiment is not None and hasattr(experiment, "node"):
            experiment.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
