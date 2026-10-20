# UR5e Robotiq pick-and-place in MuJoCo

The MJCF in `src/ur5e_mujoco_sim/models/ur5e_robotiq_scene.xml` contains the
UR5e, Robotiq gripper, table, five colored cubes, and enlarged blue tray.
`mujoco_ros2_control` exposes the simulated arm and gripper to `ros2_control`;
MoveIt 2 plans collision-aware arm trajectories. The task reads each cube's
live MuJoCo pose, picks and places the cubes one at a time, verifies the
released pose, and returns the arm home after all five are in the tray.

## Build

```bash
cd /home/dinhsieu/ur5e_pick_place_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select ur5e_mujoco_sim
source install/setup.bash
```

## Run

Terminal 1 starts MuJoCo, MoveIt, and the controllers:

```bash
cd /home/dinhsieu/ur5e_pick_place_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur5e_mujoco_sim mujoco.launch.xml
```

Wait for `joint_state_broadcaster`, `ur5e_controller`, and
`gripper_controller` to become active and for `move_group` to load OMPL.
Then run the task in terminal 2:

```bash
cd /home/dinhsieu/ur5e_pick_place_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur5e_mujoco_sim pick_place
```

For each cube, the task approaches, descends to grasp, raises once to the
transport height, moves above its assigned tray slot, descends and opens the
gripper, then raises once to clear the tray before continuing to the next cube.
It verifies each release from `/mujoco/free_joint_states` and returns home after
all five placements.

Descent, lift, tray transport, and retreat use collision-checked Cartesian
paths that preserve the grasp orientation. If MoveIt returns a partial path,
the task executes that segment and continues from its endpoint, rather than
requiring the entire path to succeed in one request. Wrist joints may move as
required by the UR5e kinematics, while Cartesian continuation avoids a sudden
wrist reconfiguration at a pose-planning fallback.

Check controller state and the cube pose topic with:

```bash
ros2 control list_controllers -c /controller_manager
ros2 topic echo /joint_states --once
ros2 topic echo /mujoco/free_joint_states
```

## Scene coordinates

The tabletop top is at `z=0.45 m`. Five cubes with 6 cm sides start near
`x=0.40–0.60 m`, `y=-0.30–-0.18 m`, centered at `z=0.48 m`. The blue tray is
centered at `(0.50, 0.15, 0.45) m`; its outer footprint is `0.50 x 0.36 m`,
its wall tops are at `z=0.55 m`, and its floor top is `z=0.47 m`. The robot base
is at `(0, 0, 0.45) m`, with the same 180-degree yaw in URDF and MuJoCo.
`tool0` is aligned to the Robotiq pinch site. The task transports each cube
with its center at `z=0.63 m`, which clears the tray walls at `z=0.55 m`.
