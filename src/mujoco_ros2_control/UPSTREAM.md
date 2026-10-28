# Workspace shutdown fix

This directory vendors the core `mujoco_ros2_control` package from
[ros-controls/mujoco_ros2_control, tag 0.1.2](https://github.com/ros-controls/mujoco_ros2_control/tree/178b1c39e5010185116bc89b4e94cbb25b825f61/mujoco_ros2_control),
commit `178b1c39e5010185116bc89b4e94cbb25b825f61`.
It matches the 0.1.2 headers installed with ROS 2 Humble on this workstation.
The upstream Apache-2.0 license and source notices are preserved.

The lifecycle changes are:

1. Stop and join the ROS executor before cleaning plugin timers/services.
2. Stop physics/render threads before releasing plugin instances or model data.
3. Declare both plugin loaders before the resources that use plugin code, so
   the loaders are destroyed last. ROS callback groups can retain weak pointer
   control blocks whose deleters reside in plugin shared libraries, even after
   the plugin instances have been cleared.

The optional lodepng fetch is pinned to tested revision
`e584a8490db1b76b81bf298c11c09d3b55d9c270` when the MuJoCo vendor does not
export a revision, instead of downloading a moving HEAD.

The original installed binary was reproduced under GDB with a real wrist-camera
scene and an X11 window-close event. Dynamic-loader tracing showed
`libmujoco_ros2_control_plugins_impl.so` being finalized before the crash in
`rclcpp::CallbackGroup::~CallbackGroup()`.

Build the workspace with `--packages-up-to ur5e_mujoco_sim` so this package is
built before the application. Sourcing the workspace selects this overlay;
the ROS installation under `/opt/ros/humble` is left untouched.
