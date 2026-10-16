#!/usr/bin/env python3
"""Move the MuJoCo red cube into the blue tray using MoveIt and ros2_control."""
import math
import time
import traceback

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    MoveItErrorCodes,
    JointConstraint,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetStateValidity,
)
from mujoco_ros2_control_msgs.msg import FreeJointStateArray
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from shape_msgs.msg import SolidPrimitive


TABLE_TOP_Z = 0.45
CUBE_SIZE = 0.06
PICK_XY = (0.50, -0.25)
TRAY_XY = (0.50, 0.15)
APPROACH_CLEARANCE = 0.15
PLACE_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.02
LIFT_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.15
HOME_JOINTS = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}
GROUP_NAME = "ur5e"
TOOL_LINK = "tool0"
BOX_NAME = "red_box"
FREE_JOINT_STATES_TOPIC = "/mujoco/free_joint_states"

# These represent the same top-down grasp rotated around world Z. The cube is
# square, so each yaw is geometrically equivalent and can give IK alternatives.
GRASP_ORIENTATIONS = (
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.70710678, 0.70710678, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -0.70710678, 0.70710678, 0.0),
)


class PickPlace:
    def __init__(self):
        self.node = Node(
            "ur5e_pick_place",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.box_state = None
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
        self.get_cartesian_path = self.node.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )
        self.get_planning_scene = self.node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.get_state_validity = self.node.create_client(
            GetStateValidity, "/check_state_validity"
        )
        self.apply_planning_scene = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

        self._wait_for_server(self.arm_action, "arm trajectory action", 30.0)
        self._wait_for_server(self.gripper_action, "gripper action", 30.0)
        self._wait_for_service(self.get_motion_plan, "MoveIt plan service", 30.0)
        self._wait_for_service(self.get_cartesian_path, "MoveIt Cartesian path service", 30.0)
        self._wait_for_service(self.get_planning_scene, "MoveIt planning scene service", 30.0)
        self._wait_for_service(self.get_state_validity, "MoveIt state validity service", 30.0)
        self._wait_for_service(self.apply_planning_scene, "MoveIt scene update service", 30.0)

        self.configure_allowed_collisions()
        self.add_static_scene_to_moveit()

    @staticmethod
    def _wait_for_server(client, label, timeout):
        if not client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Timed out waiting for {label}")

    @staticmethod
    def _wait_for_service(client, label, timeout):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Timed out waiting for {label}")

    def _wait_for_future(self, future, label, timeout):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if not future.done():
            raise RuntimeError(f"Timed out waiting for {label}")
        result = future.result()
        if result is None:
            raise RuntimeError(f"{label} returned no result")
        return result

    def _call_service(self, client, request, label, timeout=30.0):
        return self._wait_for_future(client.call_async(request), label, timeout)

    def box_state_callback(self, message):
        for free_joint in message.free_joints:
            if free_joint.name != BOX_NAME:
                continue
            pose = free_joint.pose.pose
            twist = free_joint.twist.twist
            self.box_state = {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
                "orientation": (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ),
                "linear_speed": math.sqrt(
                    twist.linear.x**2 + twist.linear.y**2 + twist.linear.z**2
                ),
                "angular_speed": math.sqrt(
                    twist.angular.x**2 + twist.angular.y**2 + twist.angular.z**2
                ),
            }
            return

    def wait_for_box_stable(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        previous_position = None
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            state = self.box_state
            if state is None:
                continue
            position = (state["x"], state["y"], state["z"])
            if previous_position is None:
                previous_position = position
                continue
            displacement = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(position, previous_position))
            )
            if (
                displacement < 0.001
                and state["linear_speed"] < 0.02
                and state["angular_speed"] < 0.05
            ):
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 0.25:
                    return state
            else:
                stable_since = None
            previous_position = position
        if self.box_state is None:
            raise RuntimeError(
                f"No {BOX_NAME} state on {FREE_JOINT_STATES_TOPIC}; "
                "check the MuJoCo free-joint publisher"
            )
        raise RuntimeError(
            f"MuJoCo object {BOX_NAME} did not settle; last pose was "
            f"({self.box_state['x']:.3f}, {self.box_state['y']:.3f}, "
            f"{self.box_state['z']:.3f}) m"
        )

    @staticmethod
    def box_collision(object_id, center, size, orientation=(0.0, 0.0, 0.0, 1.0)):
        collision = CollisionObject()
        collision.header.frame_id = "world"
        collision.id = object_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = center
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = orientation
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD
        return collision

    def configure_allowed_collisions(self):
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        response = self._call_service(
            self.get_planning_scene, request, "current MoveIt collision matrix"
        )
        matrix = response.scene.allowed_collision_matrix
        names = matrix.entry_names
        rows = matrix.entry_values
        while len(rows) < len(names):
            row = AllowedCollisionEntry()
            row.enabled = [False] * len(names)
            rows.append(row)
        for row in rows:
            row.enabled = (list(row.enabled) + [False] * len(names))[: len(names)]

        def ensure_link(link_name):
            if link_name in names:
                return names.index(link_name)
            old_size = len(names)
            names.append(link_name)
            for row in rows:
                row.enabled.append(False)
            row = AllowedCollisionEntry()
            row.enabled = [False] * (old_size + 1)
            rows.append(row)
            return old_size

        def allow_pair(first, second):
            a = ensure_link(first)
            b = ensure_link(second)
            rows[a].enabled[b] = True
            rows[b].enabled[a] = True

        allow_pair("main_table", "base_link")
        allow_pair(BOX_NAME, "left_finger")
        allow_pair(BOX_NAME, "right_finger")
        # The simplified URDF housing box overlaps the wrist_1 collision proxy
        # in the folded carry pose. MuJoCo's actual gripper mesh has clearance;
        # ignore this proxy-only pair so MoveIt can plan the carry to the tray.
        allow_pair("wrist_1_link", "gripper_base_link")
        # The grasped cube starts in contact with the tabletop. MoveIt checks
        # that attached geometry at the first lift waypoint, so explicitly
        # allow this intended support contact. At placement the cube must be
        # able to reach the tray floor before the gripper releases it.
        allow_pair(BOX_NAME, "main_table")
        allow_pair(BOX_NAME, "blue_tray_floor")

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = matrix
        apply = ApplyPlanningScene.Request()
        apply.scene = scene
        response = self._call_service(
            self.apply_planning_scene, apply, "allowed collision matrix update"
        )
        if not response.success:
            raise RuntimeError("MoveIt rejected the allowed collision matrix")

    def apply_world_objects(self, objects):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call_service(
            self.apply_planning_scene, request, "planning scene update"
        )
        if not response.success:
            raise RuntimeError("MoveIt rejected a collision object update")

    def add_static_scene_to_moveit(self):
        objects = [
            # Table MJCF half-sizes (0.8, 0.7, 0.25); MoveIt needs full sizes.
            self.box_collision("main_table", (0.3, 0.0, 0.2), (1.6, 1.4, 0.5)),
            # Tray center (0.50, 0.15), floor top z=0.47.
            self.box_collision("blue_tray_floor", (0.50, 0.15, 0.46), (0.18, 0.18, 0.02)),
            self.box_collision("blue_tray_wall_x_pos", (0.58, 0.15, 0.50), (0.02, 0.18, 0.10)),
            self.box_collision("blue_tray_wall_x_neg", (0.42, 0.15, 0.50), (0.02, 0.18, 0.10)),
            self.box_collision("blue_tray_wall_y_pos", (0.50, 0.23, 0.50), (0.14, 0.02, 0.10)),
            self.box_collision("blue_tray_wall_y_neg", (0.50, 0.07, 0.50), (0.14, 0.02, 0.10)),
        ]
        self.apply_world_objects(objects)

    def update_box_in_planning_scene(self, state):
        self.apply_world_objects(
            [
                self.box_collision(
                    BOX_NAME,
                    (state["x"], state["y"], state["z"]),
                    (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    state["orientation"],
                )
            ]
        )

    def attach_box_to_tool(self):
        attached = AttachedCollisionObject()
        attached.link_name = TOOL_LINK
        attached.object.id = BOX_NAME
        attached.object.header.frame_id = TOOL_LINK
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]
        pose = Pose()
        pose.orientation.w = 1.0
        attached.object.primitives.append(primitive)
        attached.object.primitive_poses.append(pose)
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = [TOOL_LINK, "gripper_base_link", "left_finger", "right_finger"]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call_service(self.apply_planning_scene, request, "attach cube")
        if not response.success:
            raise RuntimeError("MoveIt failed to attach the red cube")

    def detach_box_in_planning_scene(self, state):
        attached = AttachedCollisionObject()
        attached.link_name = TOOL_LINK
        attached.object.id = BOX_NAME
        attached.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call_service(self.apply_planning_scene, request, "detach cube")
        if not response.success:
            raise RuntimeError("MoveIt failed to detach the red cube")
        self.update_box_in_planning_scene(state)

    @staticmethod
    def make_pose(position, orientation):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = orientation
        return pose

    @staticmethod
    def pose_constraints(goal):
        constraints = Constraints()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.015]
        sphere_pose = Pose()
        sphere_pose.position = goal.pose.position
        sphere_pose.orientation.w = 1.0
        region = BoundingVolume()
        region.primitives.append(sphere)
        region.primitive_poses.append(sphere_pose)
        position = PositionConstraint()
        position.header = goal.header
        position.link_name = TOOL_LINK
        position.constraint_region = region
        position.weight = 1.0
        constraints.position_constraints.append(position)
        orientation = OrientationConstraint()
        orientation.header = goal.header
        orientation.link_name = TOOL_LINK
        orientation.orientation = goal.pose.orientation
        orientation.absolute_x_axis_tolerance = 0.08
        orientation.absolute_y_axis_tolerance = 0.08
        orientation.absolute_z_axis_tolerance = 0.08
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)
        return constraints

    def plan_to_pose(self, position, stage, try_yaws=False):
        orientations = GRASP_ORIENTATIONS if try_yaws else (self.grasp_orientation,)
        for orientation in orientations:
            goal = self.make_pose(position, orientation)
            request = GetMotionPlan.Request()
            plan_request = request.motion_plan_request
            plan_request.group_name = GROUP_NAME
            # Empty pipeline_id uses move_group's configured default OMPL pipeline.
            plan_request.start_state.is_diff = True
            plan_request.goal_constraints = [self.pose_constraints(goal)]
            plan_request.num_planning_attempts = 5
            plan_request.allowed_planning_time = 5.0
            plan_request.max_velocity_scaling_factor = 0.05
            plan_request.max_acceleration_scaling_factor = 0.05
            response = self._call_service(
                self.get_motion_plan, request, f"MoveIt {stage} plan", timeout=20.0
            ).motion_plan_response
            if response.error_code.val == MoveItErrorCodes.SUCCESS and response.trajectory.joint_trajectory.points:
                self.execute_arm_trajectory(response.trajectory.joint_trajectory, stage)
                if try_yaws:
                    self.grasp_orientation = orientation
                return
            self.node.get_logger().warning(
                f"MoveIt could not plan {stage} with orientation {orientation}; "
                f"error code {response.error_code.val}"
            )
        raise RuntimeError(f"MoveIt found no valid {stage} trajectory")

    def plan_to_home(self):
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = PlanningSceneComponents.ROBOT_STATE
        scene = self._call_service(
            self.get_planning_scene, scene_request, "current joint state for home"
        ).scene
        joint_state = scene.robot_state.joint_state
        current = dict(zip(joint_state.name, joint_state.position))

        request = GetMotionPlan.Request()
        plan_request = request.motion_plan_request
        plan_request.group_name = GROUP_NAME
        plan_request.start_state.is_diff = True
        constraints = Constraints()
        targets = {}
        for joint_name, home_position in HOME_JOINTS.items():
            if joint_name not in current:
                raise RuntimeError(
                    f"Current robot state has no position for {joint_name}"
                )
            # UR joints can represent the same orientation after +/- 2*pi.
            # Select the equivalent home angle requiring the least motion,
            # while staying inside the URDF's +/- 2*pi bounds.
            candidates = [
                home_position + 2.0 * math.pi * turns
                for turns in range(-2, 3)
                if -2.0 * math.pi
                <= home_position + 2.0 * math.pi * turns
                <= 2.0 * math.pi
            ]
            position = min(candidates, key=lambda value: abs(value - current[joint_name]))
            targets[joint_name] = position
            joint = JointConstraint()
            joint.joint_name = joint_name
            joint.position = position
            joint.tolerance_above = 0.05
            joint.tolerance_below = 0.05
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        plan_request.goal_constraints = [constraints]
        plan_request.num_planning_attempts = 5
        plan_request.allowed_planning_time = 5.0
        plan_request.max_velocity_scaling_factor = 0.02
        plan_request.max_acceleration_scaling_factor = 0.02
        self.node.get_logger().info(
            "Planning shortest equivalent home joint target: "
            + ", ".join(f"{name}={value:.3f}" for name, value in targets.items())
        )
        response = self._call_service(
            self.get_motion_plan, request, "MoveIt home plan", timeout=20.0
        ).motion_plan_response
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not response.trajectory.joint_trajectory.points
        ):
            raise RuntimeError(
                f"MoveIt found no valid home trajectory; error code "
                f"{response.error_code.val}"
            )
        self.execute_arm_trajectory(response.trajectory.joint_trajectory, "home")

    def plan_cartesian_segment(self, target_position, stage):
        goal = self.make_pose(target_position, self.grasp_orientation)
        request = GetCartesianPath.Request()
        request.header = goal.header
        request.start_state.is_diff = True
        request.group_name = GROUP_NAME
        request.link_name = TOOL_LINK
        request.waypoints = [goal.pose]
        request.max_step = 0.01
        request.jump_threshold = 2.0
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = 0.05
        request.max_acceleration_scaling_factor = 0.05
        response = self._call_service(
            self.get_cartesian_path, request, f"Cartesian {stage} path", timeout=20.0
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS or response.fraction < 0.99:
            raise RuntimeError(
                f"Cartesian {stage} path achieved {response.fraction:.1%}; "
                f"MoveIt error code {response.error_code.val}"
            )
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            raise RuntimeError(f"MoveIt returned an empty Cartesian {stage} trajectory")
        self.execute_arm_trajectory(trajectory, stage)

    def execute_arm_trajectory(self, trajectory, stage):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal_future = self.arm_action.send_goal_async(goal)
        goal_handle = self._wait_for_future(goal_future, f"{stage} goal acceptance", 10.0)
        if not goal_handle.accepted:
            raise RuntimeError(f"ros2_control rejected the {stage} trajectory")
        wrapped = self._wait_for_future(
            goal_handle.get_result_async(), f"{stage} trajectory", 120.0
        )
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                f"ros2_control {stage} failed: {wrapped.result.error_string}"
            )

    def set_gripper(self, opened):
        position = 0.0 if opened else 0.8
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 5.0
        self.node.get_logger().info("Opening gripper" if opened else "Closing gripper")
        handle = self._wait_for_future(
            self.gripper_action.send_goal_async(goal), "gripper goal", 10.0
        )
        if not handle.accepted:
            raise RuntimeError("Gripper controller rejected the command")
        wrapped = self._wait_for_future(handle.get_result_async(), "gripper result", 20.0)
        result = wrapped.result
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"Gripper action failed with status {wrapped.status}")
        if opened and (not result.reached_goal or result.position > 0.05):
            self.node.get_logger().warning(
                f"Gripper did not reach fully-open target (position "
                f"{result.position:.3f}, reached_goal={result.reached_goal}); "
                "checking that the cube is released in the tray before retreat"
            )
        if not opened and (result.reached_goal or not result.stalled or result.position >= 0.78):
            raise RuntimeError(
                f"Gripper closed without detecting the cube (position {result.position:.3f})"
            )

    def verify_object_pose(self, expected_xyz, label, tolerance_xy=0.04, timeout=5.0):
        state = self.wait_for_box_stable(timeout=timeout)
        dx = state["x"] - expected_xyz[0]
        dy = state["y"] - expected_xyz[1]
        dz = state["z"] - expected_xyz[2]
        if math.hypot(dx, dy) > tolerance_xy or abs(dz) > 0.04:
            raise RuntimeError(
                f"{label} verification failed; measured cube pose "
                f"({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) m"
            )
        self.node.get_logger().info(
            f"Verified {label}: ({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) m"
        )
        return state

    def run(self):
        self.set_gripper(opened=True)
        box = self.wait_for_box_stable()
        self.update_box_in_planning_scene(box)
        pick_xyz = (box["x"], box["y"], box["z"])
        approach_z = box["z"] + 0.15
        pick_approach = (pick_xyz[0], pick_xyz[1], approach_z)

        self.node.get_logger().info(
            f"PICK target from MuJoCo: ({box['x']:.3f}, {box['y']:.3f}, {box['z']:.3f}) m"
        )
        self.plan_to_pose(pick_approach, "pick approach", try_yaws=True)
        self.plan_cartesian_segment(pick_xyz, "descent")
        self.set_gripper(opened=False)
        box_after_close = self.wait_for_box_stable(timeout=3.0)
        grasp_shift = math.dist(
            (box_after_close["x"], box_after_close["y"], box_after_close["z"]),
            pick_xyz,
        )
        if grasp_shift > 0.04:
            raise RuntimeError(f"Cube shifted {grasp_shift:.3f} m while closing gripper")
        self.attach_box_to_tool()

        self.node.get_logger().info("LIFT: raise cube clear of the table")
        self.plan_cartesian_segment((pick_xyz[0], pick_xyz[1], LIFT_Z), "lift")
        self.verify_object_pose(
            (pick_xyz[0], pick_xyz[1], LIFT_Z), "lift", tolerance_xy=0.05
        )

        place_approach = (TRAY_XY[0], TRAY_XY[1], LIFT_Z)
        self.node.get_logger().info(
            f"PLACE target: tray center ({TRAY_XY[0]:.2f}, {TRAY_XY[1]:.2f}) m"
        )
        # Carry horizontally with Cartesian interpolation from the current
        # lifted pose. A fresh OMPL pose goal can choose a distant redundant
        # wrist configuration here, producing an awkward arm fold and a
        # trajectory the MuJoCo position controllers may not track well.
        self.plan_cartesian_segment(place_approach, "transport to tray")
        self.plan_cartesian_segment((TRAY_XY[0], TRAY_XY[1], PLACE_Z), "place descent")
        self.set_gripper(opened=True)
        placed = self.wait_for_box_stable(timeout=5.0)
        self.detach_box_in_planning_scene(placed)
        self.verify_object_pose((TRAY_XY[0], TRAY_XY[1], PLACE_Z), "place")
        self.node.get_logger().info("RETREAT: lift vertically clear of the tray")
        self.plan_cartesian_segment(place_approach, "retreat")
        self.node.get_logger().info("Returning arm to the SRDF home joint pose")
        self.plan_to_home()
        self.node.get_logger().info(
            "Pick-and-place completed, cube released in tray, arm returned home"
        )


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


if __name__ == "__main__":
    raise SystemExit(main())
