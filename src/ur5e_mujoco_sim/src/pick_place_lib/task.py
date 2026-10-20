"""Pick, place, and task verification flow."""
import math

from .config import (
    CUBE_SIZE,
    OBJECT_NAMES,
    PLACE_Z,
    TRAY_SLOTS,
    TRAY_XY,
    TRANSPORT_Z,
)


class PickPlaceTaskMixin:
    def verify_object_pose(
        self, object_name, expected_xyz, label, tolerance_xy=0.04, timeout=5.0
    ):
        state = self.wait_for_box_stable(object_name, timeout=timeout)
        dx = state["x"] - expected_xyz[0]
        dy = state["y"] - expected_xyz[1]
        dz = state["z"] - expected_xyz[2]
        if math.hypot(dx, dy) > tolerance_xy or abs(dz) > 0.04:
            raise RuntimeError(
                f"{label} verification failed for {object_name}; measured pose "
                f"({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) m"
            )
        self.node.get_logger().info(
            f"Verified {label}: ({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f}) m"
        )
        return state

    def process_object(self, object_name, slot_offset):
        self.set_gripper(opened=True)
        box = self.wait_for_box_stable(object_name)
        self.update_box_in_planning_scene(object_name, box)
        pick_xyz = (box["x"], box["y"], box["z"])
        approach_z = box["z"] + 0.15
        pick_approach = (pick_xyz[0], pick_xyz[1], approach_z)

        self.node.get_logger().info(
            f"PICK {object_name}: ({box['x']:.3f}, {box['y']:.3f}, "
            f"{box['z']:.3f}) m"
        )
        self.plan_to_pose(
            pick_approach,
            f"{object_name} pick approach",
            try_yaws=True,
            keep_wrist_1_outside=False,
        )
        self.plan_cartesian_to_target(
            pick_approach,
            pick_xyz,
            f"{object_name} descent",
        )
        self.set_gripper(opened=False)
        box_after_close = self.wait_for_box_stable(object_name, timeout=3.0)
        grasp_shift = math.dist(
            (box_after_close["x"], box_after_close["y"], box_after_close["z"]),
            pick_xyz,
        )
        if grasp_shift > 0.04:
            raise RuntimeError(f"Cube shifted {grasp_shift:.3f} m while closing gripper")
        self.attach_box_to_tool(object_name)

        # One lift moves directly from the grasp pose to transfer height.
        self.node.get_logger().info(
            f"LIFT {object_name}: raise once to transport height"
        )
        raise_stage = f"{object_name} lift to transport height"
        raise_xyz = (pick_xyz[0], pick_xyz[1], TRANSPORT_Z)
        self.plan_cartesian_to_target(
            pick_xyz,
            raise_xyz,
            raise_stage,
        )
        self.verify_object_pose(
            object_name,
            raise_xyz,
            f"{object_name} transport height",
            tolerance_xy=0.06,
        )

        tray_xy = (TRAY_XY[0] + slot_offset[0], TRAY_XY[1] + slot_offset[1])
        place_approach = (tray_xy[0], tray_xy[1], raise_xyz[2])
        self.node.get_logger().info(
            f"PLACE {object_name}: tray slot ({tray_xy[0]:.3f}, "
            f"{tray_xy[1]:.3f}) m"
        )

        transport_stage = f"{object_name} transport to tray"
        self.plan_cartesian_to_target(
            raise_xyz,
            place_approach,
            transport_stage,
            velocity_scale=0.04,
            acceleration_scale=0.04,
            completion_fraction=0.95,
        )

        place_xyz = (tray_xy[0], tray_xy[1], PLACE_Z)
        self.plan_cartesian_to_target(
            place_approach,
            place_xyz,
            f"{object_name} place descent",
        )
        self.set_gripper(opened=True)
        placed = self.wait_for_box_stable(object_name, timeout=5.0)
        self.detach_box_in_planning_scene(object_name, placed)
        self.verify_object_pose(object_name, place_xyz, f"{object_name} place")
        self.node.get_logger().info(f"RETREAT {object_name}: clear of the tray")
        self.plan_cartesian_to_target(
            place_xyz,
            place_approach,
            f"{object_name} retreat",
            completion_fraction=0.95,
        )

    def run(self):
        initial_states = {
            object_name: self.wait_for_box_stable(object_name)
            for object_name in OBJECT_NAMES
        }
        self.apply_world_objects(
            [
                self.box_collision(
                    object_name,
                    (
                        state["x"],
                        state["y"],
                        state["z"],
                    ),
                    (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    state["orientation"],
                )
                for object_name, state in initial_states.items()
            ]
        )

        for index, (object_name, slot_offset) in enumerate(
            zip(OBJECT_NAMES, TRAY_SLOTS), start=1
        ):
            self.node.get_logger().info(
                f"Starting object {index}/{len(OBJECT_NAMES)}: {object_name}"
            )
            self.process_object(object_name, slot_offset)

        self.node.get_logger().info("Returning arm to the SRDF home joint pose")
        self.plan_to_home()
        self.node.get_logger().info(
            f"Pick-and-place completed: all {len(OBJECT_NAMES)} objects are in "
            "the tray and the arm returned home"
        )
