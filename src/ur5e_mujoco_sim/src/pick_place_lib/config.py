"""Task geometry, joint preferences, and planning constants."""

TABLE_TOP_Z = 0.45
CUBE_SIZE = 0.06
TRAY_XY = (0.50, 0.15)
PLACE_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.02
# The cube center at 0.63 m clears the 0.55 m tray wall tops by 5 cm. This
# height is reachable with the validated single Cartesian lift from the grasp.
TRANSPORT_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.15
OBJECT_NAMES = ("red_box", "green_box", "yellow_box", "orange_box", "purple_box")
TRAY_SLOTS = (
    # Move the first slot 2 cm inward from the left tray wall. The gripper
    # needs more clearance here than the 6 cm cube itself.
    (-0.12, -0.065),
    (0.00, -0.065),
    (0.14, -0.065),
    (-0.07, 0.065),
    (0.07, 0.065),
)
HOME_JOINTS = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}
# Keep the wraparound selection consistent with both the URDF and MuJoCo
# actuator ranges. In particular, the elbow is limited to approximately +/-pi.
JOINT_LIMITS = {
    "shoulder_pan_joint": (-6.283, 6.283),
    "shoulder_lift_joint": (-6.283, 6.283),
    "elbow_joint": (-3.1415, 3.1415),
    "wrist_1_joint": (-6.283, 6.283),
    "wrist_2_joint": (-6.283, 6.283),
    "wrist_3_joint": (-6.283, 6.283),
}
GROUP_NAME = "ur5e"
TOOL_LINK = "tool0"
FREE_JOINT_STATES_TOPIC = "/mujoco/free_joint_states"
# In the home configuration wrist_1 is at -pi/2, which is the requested
# outward posture. Keep it on that negative side throughout the task.
WRIST_1_OUTER_CENTER = -1.5708
WRIST_1_OUTER_TOLERANCE = 2.0

# These represent the same top-down grasp rotated around world Z. The cube is
# square, so each yaw is geometrically equivalent and can give IK alternatives.
GRASP_ORIENTATIONS = (
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.70710678, 0.70710678, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -0.70710678, 0.70710678, 0.0),
)

# Alternate IK seeds help KDL discover both elbow branches. The first seed
# stays close to the current pose; the second is the familiar folded UR pose;
# the third explores the opposite elbow branch when it is the cleaner reach.
IK_SEED_POSTURES = (
    None,
    HOME_JOINTS,
    {
        "shoulder_pan_joint": -1.5708,
        "shoulder_lift_joint": -1.5708,
        "elbow_joint": -1.5708,
        "wrist_1_joint": 1.5708,
        "wrist_2_joint": -1.5708,
        "wrist_3_joint": 0.0,
    },
)
