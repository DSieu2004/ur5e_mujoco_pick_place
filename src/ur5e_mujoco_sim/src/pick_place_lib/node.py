#!/usr/bin/env python3
"""ROS 2 node composition and executable entry point."""
import traceback

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetPositionIK,
)
from mujoco_ros2_control_msgs.msg import FreeJointStateArray
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

from .config import FREE_JOINT_STATES_TOPIC, HOME_JOINTS, OBJECT_NAMES
from .motion import MotionMixin
from .scene import PlanningSceneMixin
from .task import PickPlaceTaskMixin
from trajectory_lib.recorder import TrajectoryRecorder
from trajectory_lib.lifecycle import init_task_context


class PickPlace(MotionMixin, PlanningSceneMixin, PickPlaceTaskMixin):
    def __init__(self, *, use_ground_truth=True):
        self.node = Node(
            "ur5e_pick_place",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.object_states = {}
        self.object_names = list(OBJECT_NAMES) if use_ground_truth else []
        self.grasp_orientation = None
        self.box_subscription = self.node.create_subscription(
            FreeJointStateArray,
            FREE_JOINT_STATES_TOPIC,
            self.box_state_callback,
            10,
        ) if use_ground_truth else None
        self.arm_action = ActionClient(
            self.node,
            FollowJointTrajectory,
            "/ur5e_controller/follow_joint_trajectory",
        )
        self.gripper_action = ActionClient(
            self.node,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
        )
        self.get_motion_plan = self.node.create_client(
            GetMotionPlan, "/plan_kinematic_path"
        )
        self.get_position_ik = self.node.create_client(
            GetPositionIK, "/compute_ik"
        )
        self.get_cartesian_path = self.node.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.get_planning_scene = self.node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.apply_planning_scene = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

        self._wait_for_server(
            self.arm_action, "/ur5e_controller/follow_joint_trajectory action", 30.0
        )
        self._wait_for_server(
            self.gripper_action, "/gripper_controller/gripper_cmd action", 30.0
        )
        self._wait_for_service(self.get_motion_plan, "MoveIt plan service", 30.0)
        self._wait_for_service(self.get_position_ik, "MoveIt IK service", 30.0)
        self._wait_for_service(self.get_cartesian_path, "MoveIt Cartesian path service", 30.0)
        self._wait_for_service(self.get_planning_scene, "MoveIt planning scene service", 30.0)
        self._wait_for_service(self.apply_planning_scene, "MoveIt scene update service", 30.0)

        self.configure_allowed_collisions()
        self.add_static_scene_to_moveit()
        self.trajectory_recorder = TrajectoryRecorder(self.node, HOME_JOINTS)


def main():
    init_task_context()
    task = None
    try:
        task = PickPlace.__new__(PickPlace)
        task.__init__()
        task.run()
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        print("Task interrupted; saving partial controller results.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if task is not None and hasattr(task, "trajectory_recorder"):
            try:
                task.trajectory_recorder.finish()
            except Exception as error:
                print(f"Could not save PID response graphs: {error}", flush=True)
        if task is not None and hasattr(task, "node"):
            task.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
