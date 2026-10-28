"""Collect visually discovered cubes without simulator object-pose access."""

import json
import math
from pathlib import Path
import traceback

from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import CollisionObject
import rclpy
from rclpy.executors import ExternalShutdownException

from pick_place_lib.config import CUBE_SIZE, PLACE_Z, TRANSPORT_Z
from pick_place_lib.node import PickPlace
from trajectory_lib.lifecycle import init_task_context
from .camera import WristCamera
from .motion import StableMotionMixin, WORK_JOINT_BOUNDS
from .geometry import (DOWN_ORIENTATION, OPTICAL_OFFSET, OVERVIEW_CAMERA_CENTER,
                       SOURCE_BOUNDS, TRAY_BOUNDS, TRAY_CENTERS,
                       free_tray_slots, inside)


class VisionPickPlace(StableMotionMixin, PickPlace):
    def __init__(self):
        # Explicitly disable the legacy MuJoCo ground-truth subscriber.
        super().__init__(use_ground_truth=False)
        self.setup_stable_motion()
        share = Path(get_package_share_directory("ur5e_mujoco_sim"))
        self.node.declare_parameter("weights", str(share / "models/box_detector.pt"))
        self.node.declare_parameter("confidence", 0.40)
        self.node.declare_parameter("device", "cpu")
        self.camera = WristCamera(
            self.node, self.node.get_parameter("weights").value,
            self.trajectory_recorder.output_dir,
            confidence=self.node.get_parameter("confidence").value,
            device=self.node.get_parameter("device").value)
        self.placed = []
        self.next_id = 0
        self.reserved = []
        self.depth_reserved = []
        self.scan_region = None

    @staticmethod
    def observation_state(box):
        return dict(zip(("x", "y", "z"), box.center), orientation=(0.0, 0.0, 0.0, 1.0))

    def observe(self, region="overview"):
        # The rotated, wrist-mounted camera sees the source and tray together.
        self.grasp_orientation = DOWN_ORIENTATION
        rotation_offset = (OPTICAL_OFFSET[0], -OPTICAL_OFFSET[1], -OPTICAL_OFFSET[2])
        tool_position = tuple(a - b for a, b in zip(OVERVIEW_CAMERA_CENTER, rotation_offset))
        if self.scan_region != region:
            self.plan_to_pose(tool_position, f"camera scan {region}")
        # Exclude images from the motion and wait for fresh, synchronized frames.
        after_ns = self.node.get_clock().now().nanoseconds + 250_000_000
        boxes = self.camera.detect(region, after_ns=after_ns)
        source = [box for box in boxes if inside(box.center[:2], SOURCE_BOUNDS)]
        if not source and self.camera.last_source_foreground >= 8:
            raise RuntimeError(
                "YOLO found no supported cubes, but depth still shows objects above the table; "
                "completion cannot be verified. "
                "Check occlusion, object sizes and detector weights.")
        self.depth_reserved = self.camera.last_depth_occupied_slots
        self.scan_region = region
        tray_count = sum(inside(box.center[:2], TRAY_BOUNDS) for box in boxes)
        self.node.get_logger().info(
            f"VISION overview: {len(source)} source boxes, {tray_count} tray boxes")
        return boxes

    def update_detected_scene(self, observations):
        objects = []
        assignments = []
        claimed = set()
        for box in observations:
            candidates = [(math.hypot(state["x"]-box.center[0], state["y"]-box.center[1]), name)
                          for name, state in self.object_states.items() if name not in claimed]
            distance, name = min(candidates, default=(math.inf, ""))
            if distance > 0.045:
                name = f"vision_box_{self.next_id:03d}"
                self.next_id += 1
                self.object_names.append(name)
            claimed.add(name)
            state = self.observation_state(box)
            self.object_states[name] = state
            objects.append(self.box_collision(name, box.center, (CUBE_SIZE,)*3))
            assignments.append((name, box))
        # Each update is based on a complete source + tray scan. Remove stale
        # world objects so a failed grasp can be observed and handled again.
        for name in set(self.object_states) - claimed:
            removal = CollisionObject()
            removal.id = name
            removal.operation = CollisionObject.REMOVE
            objects.append(removal)
            del self.object_states[name]
        self.apply_world_objects(objects)
        self.configure_allowed_collisions()
        return assignments

    def pick_into_slot(self, name, box, slot):
        self.scan_region = None
        pick = box.center
        approach = (pick[0], pick[1], pick[2]+0.15)
        lift = (pick[0], pick[1], TRANSPORT_Z)
        above_slot = (*slot, TRANSPORT_Z)
        place = (*slot, PLACE_Z)
        self.node.get_logger().info(
            f"VISION PICK {name}: ({pick[0]:.3f}, {pick[1]:.3f}, {pick[2]:.3f}) "
            f"confidence={box.confidence:.3f}; tray=({slot[0]:.3f}, {slot[1]:.3f})")
        self.set_gripper(opened=True)
        self.plan_to_pose(approach, f"{name} pick approach")
        self.plan_cartesian_to_target(approach, pick, f"{name} descent")
        # Gripper action checks stalled/reached_goal to reject an empty grasp.
        self.set_gripper(opened=False)
        self.attach_box_to_tool(name)
        self.plan_cartesian_to_target(pick, lift, f"{name} lift")
        self.plan_cartesian_to_target(lift, above_slot, f"{name} transport",
                                      velocity_scale=0.04, acceleration_scale=0.035)
        self.plan_cartesian_to_target(above_slot, place, f"{name} place descent")
        self.set_gripper(opened=True)
        # Predicted collision pose is replaced by the next visual measurement.
        state = dict(zip(("x", "y", "z"), place), orientation=(0.0, 0.0, 0.0, 1.0))
        self.detach_box_in_planning_scene(name, state)
        self.object_states[name] = state
        self.plan_cartesian_to_target(place, above_slot, f"{name} retreat",
                                      completion_fraction=0.95)

    def run(self):
        summary = {"pose_source": "YOLO RGB + aligned depth + timestamped TF",
                   "uses_simulator_object_poses": False, "placements": self.placed,
                   "source_bounds_xy": SOURCE_BOUNDS, "tray_capacity": len(TRAY_CENTERS),
                   "motion_policy": ("fixed elbow-up branch, upright Cartesian paths, "
                                     "one overview pose"),
                   "working_joint_bounds_rad": WORK_JOINT_BOUNDS,
                   "completed": False}
        path = self.trajectory_recorder.output_dir / "vision_summary.json"
        try:
            self.set_gripper(opened=True)
            observations = self.observe()
            source = [box for box in observations if inside(box.center[:2], SOURCE_BOUNDS)]
            tray = [box for box in observations if inside(box.center[:2], TRAY_BOUNDS)]
            summary["initial_source_detections"] = len(source)
            summary["initial_tray_detections"] = len(tray)
            while True:
                assignments = self.update_detected_scene(source + tray)
                source_assignments = [(name, box) for name, box in assignments
                                      if inside(box.center[:2], SOURCE_BOUNDS)]
                if not source_assignments:
                    # A second independent scan confirms completion, rather
                    # than accepting one empty set after occlusion/missed detections.
                    observations = self.observe()
                    source = [box for box in observations if inside(box.center[:2], SOURCE_BOUNDS)]
                    if source:
                        continue
                    break
                slots = free_tray_slots(tray, self.reserved + self.depth_reserved)
                if not slots:
                    raise RuntimeError(
                        f"Tray full: {len(source_assignments)} detected boxes remain on table")
                # Rear/left picks first keep access to boxes behind their neighbours.
                name, box = min(source_assignments,
                                key=lambda item: (item[1].center[1], item[1].center[0]))
                slot = slots[0]
                self.pick_into_slot(name, box, slot)
                observations = self.observe()
                source = [box for box in observations if inside(box.center[:2], SOURCE_BOUNDS)]
                tray = [box for box in observations if inside(box.center[:2], TRAY_BOUNDS)]
                matches = [b for b in tray if math.dist(b.center[:2], slot) < 0.035]
                if not matches:
                    raise RuntimeError(
                        f"Camera did not verify {name} in tray slot {slot}; stopping")
                measured = max(matches, key=lambda b: b.confidence)
                self.placed.append({"id": name, "pick_xyz": box.center,
                                    "slot": slot, "verified_xyz": measured.center})
                self.reserved.append(slot)
                self.node.get_logger().info(
                    f"VISION VERIFIED placement {len(self.placed)}: {name} at {measured.center}")
                path.write_text(json.dumps(summary, indent=2) + "\n")
            self.plan_to_home()
            summary["completed"] = True
            self.node.get_logger().info(
                f"Vision pick-and-place completed: {len(self.placed)} boxes placed; "
                "two empty source scans confirmed; arm returned home")
        except (KeyboardInterrupt, ExternalShutdownException):
            summary["interrupted"] = True
            raise
        except Exception as error:
            summary["error"] = str(error)
            raise
        finally:
            path.write_text(json.dumps(summary, indent=2) + "\n")
            (path.parent / "motion_quality.json").write_text(
                json.dumps({"joint_bounds_rad": WORK_JOINT_BOUNDS,
                            "planned_paths": self.motion_audit}, indent=2) + "\n")


def main():
    init_task_context()
    task = None
    try:
        task = VisionPickPlace.__new__(VisionPickPlace)
        task.__init__()
        task.run()
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        print("Task interrupted; saving partial controller and vision results.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if task is not None:
            if hasattr(task, "trajectory_recorder"):
                try:
                    task.trajectory_recorder.finish()
                except Exception as error:
                    print(f"Could not finalize profiler: {error}", flush=True)
            if hasattr(task, "node"):
                task.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
