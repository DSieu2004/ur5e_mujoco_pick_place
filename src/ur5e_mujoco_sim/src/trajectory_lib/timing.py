"""Read and check MoveIt's JointTrajectory time_from_start values."""


def duration_seconds(duration):
    return duration.sec + duration.nanosec * 1e-9


def describe_trajectory(trajectory, stage):
    """Return serializable waypoints; reject missing or unordered timestamps."""
    names = list(trajectory.joint_names)
    if not names or not trajectory.points:
        raise ValueError(f"{stage}: trajectory has no joints or waypoints")

    points = []
    previous_time = -1.0
    for index, point in enumerate(trajectory.points):
        seconds = duration_seconds(point.time_from_start)
        if seconds < 0 or seconds <= previous_time:
            raise ValueError(
                f"{stage}: waypoint {index} has non-increasing "
                f"time_from_start={seconds:.6f} s"
            )
        if len(point.positions) != len(names):
            raise ValueError(f"{stage}: waypoint {index} has incomplete joint positions")
        waypoint = {
            "time_from_start_s": round(seconds, 9),
            "positions_rad": dict(zip(names, point.positions)),
        }
        if len(point.velocities) == len(names):
            waypoint["velocities_rad_s"] = dict(zip(names, point.velocities))
        if len(point.accelerations) == len(names):
            waypoint["accelerations_rad_s2"] = dict(zip(names, point.accelerations))
        points.append(waypoint)
        previous_time = seconds

    if previous_time <= 0:
        raise ValueError(f"{stage}: trajectory duration must be positive")
    return {"stage": stage, "joint_names": names,
            "duration_s": round(previous_time, 9), "waypoints": points}
