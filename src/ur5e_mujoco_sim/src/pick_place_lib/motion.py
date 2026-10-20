"""MoveIt and ros2_control motion helpers."""
import math
import time
from copy import deepcopy

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PlanningSceneComponents,
)
from moveit_msgs.srv import (
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetPositionIK,
)
from shape_msgs.msg import SolidPrimitive

from .config import (
    GRASP_ORIENTATIONS,
    GROUP_NAME,
    HOME_JOINTS,
    IK_SEED_POSTURES,
    JOINT_LIMITS,
    TOOL_LINK,
    WRIST_1_OUTER_CENTER,
    WRIST_1_OUTER_TOLERANCE,
)


class MotionMixin:
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

    @staticmethod
    def make_pose(position, orientation):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
        pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = orientation
        return pose

    def current_robot_state(self):
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ROBOT_STATE
        response = self._call_service(
            self.get_planning_scene, request, "current robot state"
        )
        state = response.scene.robot_state
        joint_positions = dict(zip(state.joint_state.name, state.joint_state.position))
        missing = [name for name in HOME_JOINTS if name not in joint_positions]
        if missing:
            raise RuntimeError(
                "MoveIt current state is missing arm joints: " + ", ".join(missing)
            )
        return state, joint_positions

    def solve_ik(self, goal, seed_state, stage, seed_positions=None):
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = GROUP_NAME
        ik.ik_link_name = TOOL_LINK
        ik.pose_stamped = goal
        ik.robot_state = deepcopy(seed_state)
        if seed_positions is not None:
            seed_joint_state = ik.robot_state.joint_state
            seed_indices = {
                name: index for index, name in enumerate(seed_joint_state.name)
            }
            for name, position in seed_positions.items():
                if name in seed_indices:
                    seed_joint_state.position[seed_indices[name]] = position
        ik.avoid_collisions = True
        ik.timeout.sec = 0
        ik.timeout.nanosec = 300_000_000
        response = self._call_service(
            self.get_position_ik, request, f"KDL IK for {stage}", timeout=5.0
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        positions = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        if any(name not in positions for name in HOME_JOINTS):
            return None
        return positions

    @staticmethod
    def joint_goal_constraints(targets, tolerance=0.02):
        constraints = Constraints()
        for joint_name in HOME_JOINTS:
            joint = JointConstraint()
            joint.joint_name = joint_name
            joint.position = targets[joint_name]
            joint.tolerance_above = tolerance
            joint.tolerance_below = tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    @staticmethod
    def wrist_1_outside_constraint():
        """Keep wrist_1 on the outward side of the arm."""
        constraints = Constraints()
        joint = JointConstraint()
        joint.joint_name = "wrist_1_joint"
        joint.position = WRIST_1_OUTER_CENTER
        joint.tolerance_above = WRIST_1_OUTER_TOLERANCE
        joint.tolerance_below = WRIST_1_OUTER_TOLERANCE
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)
        return constraints

    @staticmethod
    def wrist_1_is_outside(position):
        return (
            WRIST_1_OUTER_CENTER - WRIST_1_OUTER_TOLERANCE
            <= position
            <= WRIST_1_OUTER_CENTER + WRIST_1_OUTER_TOLERANCE
        )

    @staticmethod
    def nearest_equivalent_joint_target(joint_name, target, current):
        """Choose the equivalent revolute angle closest to current within limits."""
        lower, upper = JOINT_LIMITS[joint_name]
        equivalents = [
            target + 2.0 * math.pi * turns
            for turns in range(-2, 3)
            if lower <= target + 2.0 * math.pi * turns <= upper
        ]
        if not equivalents:
            return target
        return min(equivalents, key=lambda value: abs(value - current))

    def plan_to_pose(
        self,
        position,
        stage,
        try_yaws=False,
        planning_time=5.0,
        velocity_scale=0.08,
        acceleration_scale=0.08,
        keep_wrist_1_outside=False,
    ):
        orientations = GRASP_ORIENTATIONS if try_yaws else (self.grasp_orientation,)
        seed_state, current = self.current_robot_state()
        candidates = []
        for orientation in orientations:
            goal = self.make_pose(position, orientation)
            seen_solutions = []
            for seed_posture in IK_SEED_POSTURES:
                solution = self.solve_ik(goal, seed_state, stage, seed_posture)
                if solution is None:
                    continue
                # KDL may return an equivalent angle on the far side of +/-pi.
                # Keep the same physical pose, represented nearest the measured
                # state and inside the actuator limits.
                solution = {
                    name: self.nearest_equivalent_joint_target(
                        name, solution[name], current[name]
                    )
                    for name in HOME_JOINTS
                }
                if any(
                    sum((solution[name] - previous[name]) ** 2 for name in HOME_JOINTS)
                    < 1e-4
                    for previous in seen_solutions
                ):
                    continue
                seen_solutions.append(solution)
                if (
                    keep_wrist_1_outside
                    and not self.wrist_1_is_outside(solution["wrist_1_joint"])
                ):
                    continue

                # Prefer smooth movement from the current pose, then mildly
                # favor the normal home elbow/wrist posture when candidates are
                # otherwise comparable. This avoids selecting a folded wrist
                # branch just because KDL returned it first.
                motion_cost = sum(
                    (solution[name] - current[name]) ** 2 for name in HOME_JOINTS
                )
                posture_cost = sum(
                    (solution[name] - HOME_JOINTS[name]) ** 2
                    for name in HOME_JOINTS
                )
                candidates.append(
                    (motion_cost + 0.15 * posture_cost, orientation, solution)
                )

            if not seen_solutions:
                self.node.get_logger().warning(
                    f"KDL found no collision-free IK solution for {stage} "
                    f"with orientation {orientation}"
                )

        if not candidates:
            raise RuntimeError(f"KDL found no valid IK solution for {stage}")

        # Stable tie breaking follows GRASP_ORIENTATIONS order. If the closest
        # IK branch has no collision-free path, try the next closest candidate.
        for cost, orientation, solution in sorted(candidates, key=lambda item: item[0]):
            request = GetMotionPlan.Request()
            plan_request = request.motion_plan_request
            plan_request.group_name = GROUP_NAME
            plan_request.start_state.is_diff = True
            plan_request.goal_constraints = [self.joint_goal_constraints(solution)]
            if keep_wrist_1_outside:
                plan_request.path_constraints = self.wrist_1_outside_constraint()
            plan_request.num_planning_attempts = 5
            plan_request.allowed_planning_time = planning_time
            plan_request.max_velocity_scaling_factor = velocity_scale
            plan_request.max_acceleration_scaling_factor = acceleration_scale
            response = self._call_service(
                self.get_motion_plan,
                request,
                f"MoveIt {stage} plan",
                timeout=20.0,
            ).motion_plan_response
            if (
                response.error_code.val != MoveItErrorCodes.SUCCESS
                or not response.trajectory.joint_trajectory.points
            ):
                self.node.get_logger().warning(
                    f"MoveIt could not plan {stage} for IK candidate "
                    f"{orientation}; error code {response.error_code.val}"
                )
                continue

            self.execute_arm_trajectory(response.trajectory.joint_trajectory, stage)
            if try_yaws:
                self.grasp_orientation = orientation
            self.node.get_logger().info(
                f"Selected IK for {stage}; score={cost:.4f} rad^2; "
                + ", ".join(
                    f"{name}={solution[name]:.3f}" for name in HOME_JOINTS
                )
            )
            return

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
            lower, upper = JOINT_LIMITS[joint_name]
            candidates = [
                home_position + 2.0 * math.pi * turns
                for turns in range(-2, 3)
                if lower <= home_position + 2.0 * math.pi * turns <= upper
            ]
            if not candidates:
                raise RuntimeError(
                    f"No equivalent home angle for {joint_name} is inside "
                    f"its limits [{lower:.3f}, {upper:.3f}] rad"
                )
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

    def plan_cartesian_segment(
        self,
        target_position,
        stage,
        velocity_scale=0.08,
        acceleration_scale=0.08,
        keep_wrist_1_outside=False,
        min_fraction=0.99,
        max_step=0.005,
    ):
        goal = self.make_pose(target_position, self.grasp_orientation)
        request = GetCartesianPath.Request()
        request.header = goal.header
        request.start_state.is_diff = True
        request.group_name = GROUP_NAME
        request.link_name = TOOL_LINK
        request.waypoints = [goal.pose]
        request.max_step = max_step
        request.jump_threshold = 2.0
        request.avoid_collisions = True
        if keep_wrist_1_outside:
            request.path_constraints = self.wrist_1_outside_constraint()
        request.max_velocity_scaling_factor = velocity_scale
        request.max_acceleration_scaling_factor = acceleration_scale
        response = self._call_service(
            self.get_cartesian_path, request, f"Cartesian {stage} path", timeout=20.0
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"Cartesian {stage} failed; MoveIt error code {response.error_code.val}"
            )
        if response.fraction <= 0.0:
            return 0.0
        if response.fraction < min_fraction:
            raise RuntimeError(
                f"Cartesian {stage} path achieved {response.fraction:.1%}; "
                f"required {min_fraction:.0%}; MoveIt error code "
                f"{response.error_code.val}"
            )
        trajectory = response.solution.joint_trajectory
        if not trajectory.points:
            raise RuntimeError(f"MoveIt returned an empty Cartesian {stage} trajectory")
        self.execute_arm_trajectory(trajectory, stage)
        return response.fraction

    def plan_cartesian_to_target(
        self,
        start_position,
        target_position,
        stage,
        velocity_scale=0.08,
        acceleration_scale=0.08,
        max_segments=12,
        completion_fraction=0.99,
    ):
        """Execute a Cartesian move in resumable pieces when MoveIt truncates it."""
        current_position = tuple(start_position)
        target_position = tuple(target_position)
        initial_distance = max(math.dist(current_position, target_position), 1e-9)
        cumulative_fraction = 0.0

        for segment_index in range(1, max_segments + 1):
            remaining_distance = math.dist(current_position, target_position)
            if 1.0 - remaining_distance / initial_distance >= completion_fraction:
                return

            fraction = self.plan_cartesian_segment(
                target_position,
                f"{stage} segment {segment_index}",
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                min_fraction=0.0,
                max_step=min(0.005, remaining_distance / 10.0),
            )
            if fraction <= 0.0:
                if cumulative_fraction >= completion_fraction:
                    self.node.get_logger().warning(
                        f"{stage} stopped at {cumulative_fraction:.1%} progress; "
                        "accepting the near-complete Cartesian motion"
                    )
                    return
                raise RuntimeError(
                    f"Cartesian {stage} made no progress at "
                    f"{cumulative_fraction:.1%}; tool0 is estimated to remain "
                    f"{remaining_distance:.3f} m from target"
                )

            cumulative_fraction = 1.0 - (1.0 - cumulative_fraction) * (1.0 - fraction)
            current_position = tuple(
                start + fraction * (target - start)
                for start, target in zip(current_position, target_position)
            )
            remaining_distance = math.dist(current_position, target_position)
            self.node.get_logger().info(
                f"{stage} progress: {cumulative_fraction:.1%} "
                f"(Cartesian fraction {fraction:.1%})"
            )
            if fraction >= 0.999 or cumulative_fraction >= completion_fraction:
                return
            if fraction < 0.001:
                raise RuntimeError(
                    f"Cartesian {stage} made no measurable progress; "
                    f"tool0 remains {remaining_distance:.3f} m from target"
                )

        remaining_distance = math.dist(current_position, target_position)
        raise RuntimeError(
            f"Cartesian {stage} stopped {remaining_distance:.3f} m from its target "
            f"after {max_segments} continuation segments"
        )

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
