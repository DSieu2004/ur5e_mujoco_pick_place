"""Check camera configuration before starting simulation; never read object poses."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import OPTICAL_OFFSET, OPTICAL_ROTATION


def quaternion_rotation(values):
    q = np.asarray(values, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def rpy_rotation(values):
    roll, pitch, yaw = values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                     (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
                     (-sp, cp*sr, cp*cr)))


def mjcf_transform(element, degrees):
    matrix = np.eye(4)
    matrix[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    if "quat" in element.attrib:
        matrix[:3, :3] = quaternion_rotation(np.fromstring(element.get("quat"), sep=" "))
    elif "euler" in element.attrib:
        angles = np.fromstring(element.get("euler"), sep=" ")
        if degrees:
            angles = np.radians(angles)
        if np.count_nonzero(abs(angles) > 1e-12) > 1:
            raise ValueError("Use explicit quaternions for multi-axis MJCF camera calibration")
        matrix[:3, :3] = rpy_rotation(angles)
    elif any(key in element.attrib for key in ("axisangle", "xyaxes", "zaxis")):
        raise ValueError("Use explicit pos/quat for the wrist-camera calibration")
    return matrix


def validate_camera_mount(scene_path, description_path):
    """Reject cached scenes whose physical camera disagrees with the robot TF."""
    root = ET.parse(Path(scene_path)).getroot()
    compiler = root.find("compiler")
    degrees = compiler is None or compiler.get("angle", "degree") != "radian"
    found = {}
    parents = {child: parent for parent in root.iter() for child in parent}
    required = set()
    for target in (root.find(".//camera[@name='wrist_camera']"),
                   root.find(".//site[@name='ur5e/robotiq_2f85/pinch']")):
        while target is not None:
            required.add(target)
            target = parents.get(target)

    def walk(element, parent):
        transform = parent
        if element.tag in ("body", "camera", "site"):
            transform = parent @ mjcf_transform(element, degrees)
        if element.tag == "camera" and element.get("name") == "wrist_camera":
            found["camera"] = transform
        if element.tag == "site" and element.get("name") == "ur5e/robotiq_2f85/pinch":
            found["tool"] = transform
        for child in element:
            if child in required and child.tag in ("body", "camera", "site"):
                walk(child, transform)
    walk(root.find("worldbody"), np.eye(4))
    if set(found) != {"camera", "tool"}:
        raise ValueError("Scene must contain the wrist_camera and the Robotiq pinch site")
    optical = np.linalg.inv(found["tool"]) @ found["camera"]
    # OpenGL camera (-Z forward, +Y up) to ROS optical (+Z forward, +Y down).
    optical[:3, :3] = optical[:3, :3] @ np.diag((1., -1., -1.))
    expected = np.eye(4)
    expected[:3, 3] = OPTICAL_OFFSET
    expected[:3, :3] = OPTICAL_ROTATION
    description = ET.parse(Path(description_path)).getroot()
    urdf = np.eye(4)
    for name in ("tool0_to_wrist_camera", "wrist_camera_optical_joint"):
        origin = description.find(f"./joint[@name='{name}']/origin")
        if origin is None:
            raise ValueError(f"Missing camera joint {name} in robot description")
        local = np.eye(4)
        local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        local[:3, :3] = rpy_rotation(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        urdf = urdf @ local
    if not np.allclose(optical, expected, atol=1e-6) or not np.allclose(urdf, expected, atol=1e-6):
        raise ValueError(
            "Scene camera mount and optical TF do not match this project version. "
            "Regenerate the scene with tools/generate_box_scene.py before running the task.")
    return optical
