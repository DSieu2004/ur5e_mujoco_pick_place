# UR5e Robotiq pick-and-place in MuJoCo

The MJCF in `src/ur5e_mujoco_sim/models/ur5e_robotiq_scene.xml` contains the
UR5e, Robotiq gripper, table, red cube, and blue tray. `mujoco_ros2_control`
opens MuJoCo and exposes the arm and gripper to `ros2_control`. `move_group`
plans the free-space approach and collision-checked Cartesian pick/place
segments. The task verifies the cube's MuJoCo free-joint pose after lifting
and after release.

The cube and tray centers are now separated by 40 cm: cube `(0.50, -0.25) m`,
tray `(0.50, 0.15) m`. Both remain within the table bounds and near the robot.

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

The task opens the gripper, reads the live cube pose, approaches and grasps the
cube, lifts it, carries it to the blue tray, lowers it, releases it, and
verifies the resulting MuJoCo pose.

Check controller state and the cube pose topic with:

```bash
ros2 control list_controllers -c /controller_manager
ros2 topic echo /joint_states --once
ros2 topic echo /mujoco/free_joint_states
```

## Scene coordinates

The tabletop top is at `z=0.45 m`. The red cube starts centered at
`(0.50, -0.25, 0.48) m` and has 6 cm sides. The tray is centered at
`(0.50, 0.15, 0.45) m`; its floor top is `z=0.47 m`. The robot base is at
`(0, 0, 0.45) m`, with the same 180-degree yaw in URDF and MuJoCo. `tool0` is
aligned to the Robotiq pinch site.
