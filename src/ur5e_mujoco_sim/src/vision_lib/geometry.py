"""Camera geometry and tray allocation, independent of ROS and MuJoCo state."""

from dataclasses import dataclass

import numpy as np


TABLE_Z = 0.45
CUBE_SIZE = 0.06
SOURCE_BOUNDS = (0.25, 0.75, -0.46, -0.10)
TRAY_BOUNDS = (0.30, 0.70, 0.025, 0.275)
TRAY_CENTERS = tuple((x, y) for y in (0.085, 0.215) for x in (0.40, 0.49, 0.58, 0.67))
# The optical frame is a rigid transform from tool0, shared with MJCF/URDF.
OPTICAL_OFFSET = np.array((0.068, 0.0325, -0.144))
OPTICAL_ROTATION = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
OVERVIEW_CAMERA_CENTER = np.array((0.50, -0.0925, 1.20))
DOWN_ORIENTATION = (0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class BoxObservation:
    center: tuple
    confidence: float
    bbox: tuple


def inside(xy, bounds):
    x0, x1, y0, y1 = bounds
    return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1


def projection_matrix(intrinsics):
    """Reject uncalibrated CameraInfo instead of treating invalid depth as empty."""
    matrix = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera intrinsics must be finite with positive fx and fy")
    return matrix


def deproject_box(depth, intrinsics, world_from_optical, bbox, confidence):
    """
    Estimate a cube centre from the visible top surface, not bbox centre depth.

    Depth is aligned to RGB and expressed along optical +Z in metres. The
    table/tray heights distinguish cube top pixels from support/background.
    """
    height, width = depth.shape
    x0, y0, x1, y1 = np.asarray(bbox, dtype=int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    v, u = np.mgrid[y0:y1:2, x0:x1:2]
    z = depth[v, u]
    valid = np.isfinite(z) & (z >= 0.28) & (z <= 1.5)
    if valid.sum() < 15:
        return None
    k = projection_matrix(intrinsics)
    points = np.stack(((u[valid] - k[0, 2]) * z[valid] / k[0, 0],
                       (v[valid] - k[1, 2]) * z[valid] / k[1, 1], z[valid]), axis=1)
    transform = np.asarray(world_from_optical)
    world = points @ transform[:3, :3].T + transform[:3, 3]
    # Both table and tray boxes are 6 cm tall; their support differs by 2 cm.
    results = []
    for bounds, support in ((SOURCE_BOUNDS, TABLE_Z), (TRAY_BOUNDS, TABLE_Z + 0.02)):
        top = support + CUBE_SIZE
        mask = ((abs(world[:, 2] - top) < 0.009)
                & (world[:, 0] >= bounds[0] - CUBE_SIZE / 2)
                & (world[:, 0] <= bounds[1] + CUBE_SIZE / 2)
                & (world[:, 1] >= bounds[2] - CUBE_SIZE / 2)
                & (world[:, 1] <= bounds[3] + CUBE_SIZE / 2))
        surface = world[mask]
        if len(surface) < 15:
            continue
        # Robust bounding centre avoids bias towards the visible side face.
        lower, upper = np.quantile(surface[:, :2], (0.03, 0.97), axis=0)
        extent = upper - lower
        if np.any(extent < 0.022) or np.any(extent > 0.095):
            continue
        xy = (lower + upper) / 2
        if inside(xy, bounds):
            results.append((len(surface), BoxObservation(
                (float(xy[0]), float(xy[1]), support + CUBE_SIZE / 2),
                float(confidence), tuple(float(v) for v in bbox))))
    return max(results, key=lambda item: item[0])[1] if results else None


def merge_observations(observations, distance=0.035):
    merged = []
    for observation in sorted(observations, key=lambda box: -box.confidence):
        if all(np.linalg.norm(np.subtract(observation.center[:2], other.center[:2])) > distance
               for other in merged):
            merged.append(observation)
    return merged


def persistent_observations(frames, min_frames=2, distance=0.035):
    """Require support from distinct RGB-D frames, regardless of duplicate boxes."""
    candidates = merge_observations([box for frame in frames for box in frame], distance)
    return [box for box in candidates if sum(
        any(np.linalg.norm(np.subtract(box.center[:2], other.center[:2])) < distance
            for other in frame) for frame in frames) >= min_frames]


def sampled_world_points(depth, intrinsics, world_from_optical):
    v, u = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
    z = depth[v, u]
    valid = np.isfinite(z) & (z >= 0.28) & (z <= 1.5)
    k = projection_matrix(intrinsics)
    if not valid.any():
        raise ValueError("RGB-D frame has no usable depth; an empty table cannot be verified")
    points = np.stack(((u[valid]-k[0, 2])*z[valid]/k[0, 0],
                       (v[valid]-k[1, 2])*z[valid]/k[1, 1], z[valid]), axis=1)
    transform = np.asarray(world_from_optical)
    return points @ transform[:3, :3].T + transform[:3, 3]


def source_foreground_count(depth, intrinsics, world_from_optical):
    """Veto completion when elevated depth remains; never create grasp targets."""
    world = sampled_world_points(depth, intrinsics, world_from_optical)
    x0, x1, y0, y1 = SOURCE_BOUNDS
    mask = ((world[:, 0] >= x0) & (world[:, 0] <= x1)
            & (world[:, 1] >= y0) & (world[:, 1] <= y1)
            & (world[:, 2] > TABLE_Z + 0.035) & (world[:, 2] < TABLE_Z + 0.30))
    return int(np.count_nonzero(mask))


def depth_occupied_slots(depth, intrinsics, world_from_optical):
    """Conservatively block a tray footprint even when YOLO missed its occupant."""
    world = sampled_world_points(depth, intrinsics, world_from_optical)
    elevated = world[(world[:, 2] > TABLE_Z + 0.02 + 0.035)
                     & (world[:, 2] < TABLE_Z + 0.30)]
    return [slot for slot in TRAY_CENTERS if np.count_nonzero(
        (abs(elevated[:, 0] - slot[0]) < 0.04)
        & (abs(elevated[:, 1] - slot[1]) < 0.04)) >= 8]


def free_tray_slots(observations, reserved=()):
    """Reject slots whose cube/gripper footprint overlaps an observed cube."""
    occupied = [box.center[:2] for box in observations if inside(box.center[:2], TRAY_BOUNDS)]
    occupied.extend(reserved)
    return [slot for slot in TRAY_CENTERS
            if all(not (abs(slot[0] - xy[0]) < 0.085 and abs(slot[1] - xy[1]) < 0.095)
                   for xy in occupied)]
