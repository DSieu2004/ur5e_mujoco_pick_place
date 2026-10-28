"""Upright Cartesian motion on one arm IK branch, without random joint detours."""

from copy import deepcopy
import math

from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, OrientationConstraint
from moveit_msgs.srv import GetCartesianPath, GetPositionFK

from pick_place_lib.config import GROUP_NAME, HOME_JOINTS, TOOL_LINK
from pick_place_lib.motion import CartesianPathBlocked
from .geometry import DOWN_ORIENTATION


# One elbow-up, wrist-down branch, including the canonical SRDF home pose.
# Bounds prevent an equivalent 2*pi wrist winding or an elbow/wrist flip.
WORK_JOINT_BOUNDS = {
    "shoulder_pan_joint": (-4.70, -1.20),
    "shoulder_lift_joint": (-2.60, -0.50),
    "elbow_joint": (0.35, 2.90),
    "wrist_1_joint": (-2.90, -0.50),
    "wrist_2_joint": (-2.00, -1.10),
    "wrist_3_joint": (-1.90, 1.90),
}
MAX_WAYPOINT_STEP_RAD = 0.12


def work_constraints():
    constraints = Constraints()
    for name, (lower, upper) in WORK_JOINT_BOUNDS.items():
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = (lower + upper) / 2
        joint.tolerance_above = joint.tolerance_below = (upper - lower) / 2
        joint.weight = 1.0
        constraints.joint_constraints.append(joint)
    orientation = OrientationConstraint()
    orientation.header.frame_id = "world"
    orientation.link_name = TOOL_LINK
    orientation.orientation.x = 1.0
    orientation.orientation.w = 0.0
    orientation.absolute_x_axis_tolerance = 0.075
    orientation.absolute_y_axis_tolerance = 0.075
    orientation.absolute_z_axis_tolerance = math.pi
    orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
    orientation.weight = 1.0
    constraints.orientation_constraints.append(orientation)
    return constraints


def inspect_joint_path(trajectory):
    """Check the complete path before any waypoint is executed."""
    if set(trajectory.joint_names) != set(WORK_JOINT_BOUNDS):
        raise CartesianPathBlocked("Cartesian path is missing arm joints")
    if len(trajectory.points) < 2:
        raise CartesianPathBlocked("Cartesian path contains no timed motion")
    values = [list(point.positions) for point in trajectory.points]
    count = len(trajectory.joint_names)
    if any(len(row) != count or not all(math.isfinite(v) for v in row) for row in values):
        raise CartesianPathBlocked("Cartesian path has invalid joint coordinates")
    for row in values:
        for name, value in zip(trajectory.joint_names, row):
            lower, upper = WORK_JOINT_BOUNDS[name]
            if not lower - 0.02 <= value <= upper + 0.02:
                raise CartesianPathBlocked(f"{name} leaves the working posture: {value:.3f} rad")
    steps = [[abs(b-a) for a, b in zip(first, second)]
             for first, second in zip(values, values[1:])]
    maximum = max(max(row) for row in steps)
    if maximum > MAX_WAYPOINT_STEP_RAD:
        raise CartesianPathBlocked(f"Cartesian IK jump {maximum:.3f} rad exceeds the motion limit")
    travel = (sum(row[i] for row in steps) for i in range(count))
    return {"max_waypoint_step_rad": maximum,
            "joint_travel_rad": dict(zip(trajectory.joint_names, travel))}


class StableMotionMixin:
    def setup_stable_motion(self):
        self.get_position_fk = self.node.create_client(GetPositionFK, "/compute_fk")
        self._wait_for_service(self.get_position_fk, "MoveIt FK service", 30.0)
        self.motion_audit = []

    def fk_pose(self, state):
        request = GetPositionFK.Request()
        request.header.frame_id = "world"
        request.fk_link_names = [TOOL_LINK]
        request.robot_state = state
        response = self._call_service(self.get_position_fk, request, "tool forward kinematics")
        if response.error_code.val != MoveItErrorCodes.SUCCESS or not response.pose_stamped:
            raise RuntimeError("MoveIt could not compute the tool pose")
        return response.pose_stamped[0].pose

    def try_cartesian_route(self, poses, stage, velocity_scale, acceleration_scale):
        state, positions = self.current_robot_state()
        for name, value in positions.items():
            if name not in WORK_JOINT_BOUNDS:
                continue
            lower, upper = WORK_JOINT_BOUNDS[name]
            if not lower - 0.03 <= value <= upper + 0.03:
                raise RuntimeError(
                    f"Start posture {name}={value:.3f} is outside the stable branch; "
                    "restart a fresh simulation at home before running the task")
        request = GetCartesianPath.Request()
        request.header.frame_id = "world"
        request.start_state = state
        request.group_name = GROUP_NAME
        request.link_name = TOOL_LINK
        request.waypoints = poses
        request.max_step = 0.004
        request.jump_threshold = 2.0
        request.avoid_collisions = True
        request.path_constraints = work_constraints()
        request.max_velocity_scaling_factor = velocity_scale
        request.max_acceleration_scaling_factor = acceleration_scale
        response = self._call_service(self.get_cartesian_path, request,
                                      f"upright Cartesian {stage}", timeout=30.0)
        if response.error_code.val != MoveItErrorCodes.SUCCESS or response.fraction < 0.999:
            self.node.get_logger().info(f"{stage}: candidate route covers {response.fraction:.1%}")
            return False
        trajectory = response.solution.joint_trajectory
        try:
            audit = inspect_joint_path(trajectory)
        except CartesianPathBlocked as error:
            self.node.get_logger().info(f"{stage}: rejected route: {error}")
            return False
        self.execute_arm_trajectory(trajectory, stage)
        self.motion_audit.append(dict(stage=stage, **audit))
        return True

    def move_upright(self, position, stage, velocity_scale=0.07,
                     acceleration_scale=0.04, orientation=DOWN_ORIENTATION,
                     returning_home=False):
        target = self.make_pose(position, orientation).pose
        direct = [target]
        # Fixed clear-space routes avoid passing near the base or folding
        # through the forearm. Each entire route is checked before execution.
        if returning_home:
            routes = [direct, [self.make_pose((0.40, 0.30, 0.82), DOWN_ORIENTATION).pose, target]]
        else:
            hub = self.make_pose((0.50, -0.05, 0.78), DOWN_ORIENTATION).pose
            side = self.make_pose((0.40, 0.30, 0.82), DOWN_ORIENTATION).pose
            routes = [direct, [hub, target], [side, hub, target]]
        for index, poses in enumerate(routes):
            label = stage if index == 0 else f"{stage} via clearance route {index}"
            if self.try_cartesian_route(poses, label, velocity_scale, acceleration_scale):
                return
        raise CartesianPathBlocked(
            f"No complete upright route for {stage}; "
            "stopping before moving instead of changing IK branch")

    def plan_to_pose(self, position, stage, try_yaws=False, planning_time=5.0,
                     velocity_scale=0.07, acceleration_scale=0.04,
                     keep_wrist_1_outside=True):
        self.grasp_orientation = DOWN_ORIENTATION
        self.move_upright(position, stage, velocity_scale, acceleration_scale)

    def plan_cartesian_to_target(self, start_position, target_position, stage,
                                 velocity_scale=0.07, acceleration_scale=0.04,
                                 max_segments=12, completion_fraction=0.99):
        self.grasp_orientation = DOWN_ORIENTATION
        self.move_upright(target_position, stage, velocity_scale, acceleration_scale)

    def plan_to_home(self):
        state, _ = self.current_robot_state()
        goal = deepcopy(state)
        positions = dict(zip(goal.joint_state.name, goal.joint_state.position))
        positions.update(HOME_JOINTS)
        goal.joint_state.position = [positions[name] for name in goal.joint_state.name]
        home = self.fk_pose(goal)
        q = home.orientation
        self.move_upright((home.position.x, home.position.y, home.position.z), "home",
                          velocity_scale=0.06, acceleration_scale=0.035,
                          orientation=(q.w, q.x, q.y, q.z), returning_home=True)
        _, actual = self.current_robot_state()
        if max(abs(actual[name] - value) for name, value in HOME_JOINTS.items()) > 0.05:
            raise RuntimeError("The arm did not reach the canonical home joint posture")
