#!/usr/bin/env python3
"""Show the wrist camera and YOLO overlays; closing the viewer leaves the task running."""

import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vision_lib.camera import image_array


def main():
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    rclpy.init()
    node = Node("wrist_camera_preview")
    images = {}

    def receive(kind, message):
        images[kind] = (time.monotonic(), image_array(message))
    node.camera_subscriptions = [node.create_subscription(
        Image, topic, lambda m, kind=kind: receive(kind, m), qos_profile_sensor_data)
        for kind, topic in (("raw", "/wrist_camera/color/image_raw"),
                            ("detections", "/wrist_camera/detections/image"))]
    window = "RealSense D435i wrist camera | YOLO boxes"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            detected = images.get("detections")
            image = (detected if detected and time.monotonic()-detected[0] < 2
                     else images.get("raw"))
            if image is not None:
                cv2.imshow(window, image[1])
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
