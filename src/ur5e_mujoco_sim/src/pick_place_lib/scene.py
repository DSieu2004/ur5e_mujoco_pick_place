"""Planning-scene and MuJoCo object-state helpers."""
import math
import time

import rclpy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from shape_msgs.msg import SolidPrimitive

from .config import (
    CUBE_SIZE,
    FREE_JOINT_STATES_TOPIC,
    OBJECT_NAMES,
    TABLE_TOP_Z,
    TOOL_LINK,
    TRAY_XY,
)


class PlanningSceneMixin:
    def box_state_callback(self, message):
        for free_joint in message.free_joints:
            if free_joint.name not in OBJECT_NAMES:
                continue
            pose = free_joint.pose.pose
            twist = free_joint.twist.twist
            self.object_states[free_joint.name] = {
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

    def wait_for_box_stable(self, object_name, timeout=5.0):
        deadline = time.monotonic() + timeout
        previous_position = None
        stable_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            state = self.object_states.get(object_name)
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
        state = self.object_states.get(object_name)
        if state is None:
            raise RuntimeError(
                f"No {object_name} state on {FREE_JOINT_STATES_TOPIC}; "
                "check the MuJoCo free-joint publisher"
            )
        raise RuntimeError(
            f"MuJoCo object {object_name} did not settle; last pose was "
            f"({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) m"
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
        for object_name in OBJECT_NAMES:
            allow_pair(object_name, "left_finger")
            allow_pair(object_name, "right_finger")
        # The simplified URDF housing box overlaps the wrist_1 collision proxy
        # in the folded carry pose. MuJoCo's actual gripper mesh has clearance;
        # ignore this proxy-only pair so MoveIt can plan the carry to the tray.
        allow_pair("wrist_1_link", "gripper_base_link")
        # The grasped cube starts in contact with the tabletop. MoveIt checks
        # that attached geometry at the first lift waypoint, so explicitly
        # allow this intended support contact. At placement the cube must be
        # able to reach the tray floor before the gripper releases it.
        for object_name in OBJECT_NAMES:
            allow_pair(object_name, "main_table")
            allow_pair(object_name, "blue_tray_floor")

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
            # Enlarged tray centered at (0.50, 0.15); floor top remains z=0.47.
            self.box_collision("blue_tray_floor", (0.50, 0.15, 0.46), (0.50, 0.36, 0.02)),
            self.box_collision("blue_tray_wall_x_pos", (0.74, 0.15, 0.50), (0.02, 0.36, 0.10)),
            self.box_collision("blue_tray_wall_x_neg", (0.26, 0.15, 0.50), (0.02, 0.36, 0.10)),
            self.box_collision("blue_tray_wall_y_pos", (0.50, 0.31, 0.50), (0.46, 0.02, 0.10)),
            self.box_collision("blue_tray_wall_y_neg", (0.50, -0.01, 0.50), (0.46, 0.02, 0.10)),
        ]
        self.apply_world_objects(objects)

    def update_box_in_planning_scene(self, object_name, state):
        self.apply_world_objects(
            [
                self.box_collision(
                    object_name,
                    (state["x"], state["y"], state["z"]),
                    (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    state["orientation"],
                )
            ]
        )

    def attach_box_to_tool(self, object_name):
        attached = AttachedCollisionObject()
        attached.link_name = TOOL_LINK
        attached.object.id = object_name
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
        response = self._call_service(self.apply_planning_scene, request, f"attach {object_name}")
        if not response.success:
            raise RuntimeError(f"MoveIt failed to attach {object_name}")

    def detach_box_in_planning_scene(self, object_name, state):
        attached = AttachedCollisionObject()
        attached.link_name = TOOL_LINK
        attached.object.id = object_name
        attached.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        response = self._call_service(self.apply_planning_scene, request, f"detach {object_name}")
        if not response.success:
            raise RuntimeError(f"MoveIt failed to detach {object_name}")
        self.update_box_in_planning_scene(object_name, state)
