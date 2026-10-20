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
from rclpy.node import Node
from rclpy.parameter import Parameter

from .config import FREE_JOINT_STATES_TOPIC
from .motion import MotionMixin
from .scene import PlanningSceneMixin
from .task import PickPlaceTaskMixin


class PickPlace(MotionMixin, PlanningSceneMixin, PickPlaceTaskMixin):
    def __init__(self):
        self.node = Node(
            "ur5e_pick_place",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.object_states = {}
        self.grasp_orientation = None
        self.box_subscription = self.node.create_subscription(
            FreeJointStateArray,
            FREE_JOINT_STATES_TOPIC,
            self.box_state_callback,
            10,
        )
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

        self._wait_for_server(self.arm_action, "arm trajectory action", 30.0)
        self._wait_for_server(self.gripper_action, "gripper action", 30.0)
        self._wait_for_service(self.get_motion_plan, "MoveIt plan service", 30.0)
        self._wait_for_service(self.get_position_ik, "MoveIt IK service", 30.0)
        self._wait_for_service(self.get_cartesian_path, "MoveIt Cartesian path service", 30.0)
        self._wait_for_service(self.get_planning_scene, "MoveIt planning scene service", 30.0)
        self._wait_for_service(self.apply_planning_scene, "MoveIt scene update service", 30.0)

        self.configure_allowed_collisions()
        self.add_static_scene_to_moveit()


def main():
    rclpy.init()
    task = None
    try:
        task = PickPlace()
        task.run()
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if task is not None:
            task.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
