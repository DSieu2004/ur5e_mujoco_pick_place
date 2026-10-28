"""Synchronized RGB-D acquisition, YOLO inference and timestamped ROS TF."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import (deproject_box, depth_occupied_slots, persistent_observations,
                       source_foreground_count)


def image_array(message):
    """Respect encoding, endianness and row padding without cv_bridge ABI coupling."""
    if message.encoding in ("rgb8", "bgr8"):
        a = np.ndarray((message.height, message.width, 3), dtype=np.uint8,
                       buffer=bytes(message.data), strides=(message.step, 3, 1)).copy()
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if message.encoding == "rgb8" else a
    if message.encoding in ("32FC1", "16UC1"):
        dtype = np.dtype((">" if message.is_bigendian else "<") +
                         ("f4" if message.encoding == "32FC1" else "u2"))
        a = np.ndarray((message.height, message.width), dtype=dtype,
                       buffer=bytes(message.data), strides=(message.step, dtype.itemsize))
        return a.astype(np.float32) * (0.001 if message.encoding == "16UC1" else 1.0)
    raise ValueError(f"Unsupported camera encoding {message.encoding!r}")


def transform_matrix(transform):
    p, q = transform.translation, transform.rotation
    xyzw = np.array((q.x, q.y, q.z, q.w), dtype=float)
    if not np.all(np.isfinite(xyzw)) or np.linalg.norm(xyzw) < 1e-12:
        raise ValueError("Camera TF quaternion must be finite and nonzero")
    xyzw /= np.linalg.norm(xyzw)
    x, y, z, w = xyzw
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    matrix[:3, 3] = (p.x, p.y, p.z)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Camera TF translation must be finite")
    return matrix


class WristCamera:
    def __init__(self, node, weights, output_dir, confidence=0.40, device="cpu"):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Trained box weights missing: {weights}. Run tools/train_box_detector.py first.")
        self.node = node
        self.model = YOLO(str(weights))
        if "box" not in self.model.names.values():
            raise ValueError("Detector weights must contain the trained 'box' class")
        self.confidence = confidence
        self.device = device
        self.output_dir = Path(output_dir) / "vision"
        self.output_dir.mkdir(exist_ok=True)
        self.info = None
        self.pending = {}
        self.scan_index = 0
        self.last_source_foreground = 0
        self.last_depth_occupied_slots = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.subscriptions = [
            node.create_subscription(CameraInfo, "/wrist_camera/color/camera_info",
                                     self.on_info, qos_profile_sensor_data),
            node.create_subscription(Image, "/wrist_camera/color/image_raw",
                                     lambda m: self.on_image("color", m), qos_profile_sensor_data),
            node.create_subscription(Image, "/wrist_camera/aligned_depth_to_color/image_raw",
                                     lambda m: self.on_image("depth", m), qos_profile_sensor_data),
        ]
        self.publisher = node.create_publisher(Image, "/wrist_camera/detections/image", 1)

    def on_info(self, message):
        self.info = message

    def on_image(self, kind, message):
        key = (message.header.stamp.sec, message.header.stamp.nanosec)
        self.pending.setdefault(key, {})[kind] = message
        for old in sorted(self.pending)[:-8]:
            del self.pending[old]

    def detect(self, label, after_ns, frames=3, timeout=12.0):
        deadline = time.monotonic() + timeout
        observations_by_frame = []
        processed = set()
        foreground_counts = []
        depth_slots = []
        while len(processed) < frames and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.info is None:
                continue
            for key, pair in sorted(list(self.pending.items())):
                stamp_ns = key[0] * 1_000_000_000 + key[1]
                if key in processed or stamp_ns <= after_ns or set(pair) != {"color", "depth"}:
                    continue
                color, depth_message = pair["color"], pair["depth"]
                if color.header.frame_id != depth_message.header.frame_id:
                    raise RuntimeError("RGB and aligned depth must share an optical frame")
                if (color.width, color.height) != (depth_message.width, depth_message.height):
                    raise RuntimeError("RGB and depth dimensions differ; align depth to RGB first")
                if (color.width, color.height) != (self.info.width, self.info.height):
                    raise RuntimeError("CameraInfo resolution does not match RGB/depth")
                if self.info.header.frame_id != color.header.frame_id:
                    raise RuntimeError("CameraInfo and RGB/depth must share an optical frame")
                try:
                    transform = self.buffer.lookup_transform(
                        "world", color.header.frame_id, Time.from_msg(color.header.stamp))
                except TransformException:
                    continue
                bgr, depth = image_array(color), image_array(depth_message)
                world_from_optical = transform_matrix(transform.transform)
                foreground_counts.append(
                    source_foreground_count(depth, self.info.k, world_from_optical))
                depth_slots.extend(depth_occupied_slots(depth, self.info.k, world_from_optical))
                result = self.model.predict(bgr, conf=self.confidence, imgsz=416,
                                            device=self.device, verbose=False)[0]
                found = []
                for box in result.boxes:
                    if self.model.names[int(box.cls.item())] != "box":
                        continue
                    bounds = box.xyxy[0].cpu().numpy()
                    observation = deproject_box(depth, self.info.k, world_from_optical,
                                                bounds, float(box.conf.item()))
                    if observation is not None:
                        found.append(observation)
                # Show only detections accepted by RGB-D geometry, so a
                # rejected wall/shadow prediction is not presented as a target.
                annotated = bgr.copy()
                for box in found:
                    x0, y0, x1, y1 = (int(v) for v in box.bbox)
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 220, 0), 2)
                    cv2.putText(annotated, f"box {box.confidence:.2f}",
                                (x0, max(12, y0-5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 220, 0), 1)
                    cv2.putText(annotated, f"({box.center[0]:.3f}, {box.center[1]:.3f}) m",
                                (x0, min(color.height-4, y0+15)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (255, 255, 255), 1)
                self.save_scan(label, color, bgr, depth, annotated, found, transform)
                observations_by_frame.append(found)
                processed.add(key)
                if len(processed) >= frames:
                    break
        if len(processed) < frames:
            raise RuntimeError(
                f"RGB-D scan {label!r} timed out ({len(processed)}/{frames} frames). "
                "Check camera topics, matching timestamps and optical-frame TF.")
        # Require repeat detections; a single transient false positive cannot trigger a grasp.
        self.last_source_foreground = int(np.median(foreground_counts))
        self.last_depth_occupied_slots = [slot for slot in set(depth_slots)
                                          if depth_slots.count(slot) >= 2]
        return persistent_observations(observations_by_frame)

    def save_scan(self, label, color, bgr, depth, annotated, observations, transform):
        filename = f"{self.scan_index:04d}_{label}"
        self.scan_index += 1
        cv2.imwrite(str(self.output_dir / f"{filename}.jpg"), annotated)
        np.savez_compressed(self.output_dir / f"{filename}.npz", bgr=bgr, depth=depth,
                            k=np.asarray(self.info.k),
                            world_from_optical=transform_matrix(transform.transform))
        (self.output_dir / f"{filename}.json").write_text(json.dumps({
            "label": label, "stamp": asdict_stamp(color.header.stamp),
            "optical_frame": color.header.frame_id,
            "intrinsics": list(self.info.k),
            "world_from_optical": transform_matrix(transform.transform).tolist(),
            "detections": [asdict(box) for box in observations],
        }, indent=2) + "\n")
        message = Image()
        message.header = color.header
        message.height, message.width = annotated.shape[:2]
        message.encoding = "bgr8"
        message.step = message.width * 3
        message.data = annotated.tobytes()
        self.publisher.publish(message)


def asdict_stamp(stamp):
    return {"sec": stamp.sec, "nanosec": stamp.nanosec}
