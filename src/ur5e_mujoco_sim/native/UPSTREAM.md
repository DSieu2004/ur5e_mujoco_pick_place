# MoveGroup shutdown lifetime fix

`move_group_main.cpp` is derived from the MoveIt 2 Humble executable source,
[tag 2.5.10](https://github.com/moveit/moveit2/blob/2.5.10/moveit_ros/move_group/src/move_group.cpp),
matching the installed MoveIt libraries. Its original BSD license is preserved
at the top of the file.

The local executable retains the capability, controller-manager and planner
plugin loaders until after the ROS nodes and executors have been destroyed.
The original executable unloads plugin code while callback groups still hold
weak pointer control blocks allocated in those libraries. This was reproduced
as a crash in `rclcpp::CallbackGroup::~CallbackGroup()` on both SIGINT and
window-close shutdowns.

Planning and execution capabilities remain enabled. The ROS node name and
service/action names remain `move_group`; project launch files select
`ur5e_move_group` from this package rather than changing the ROS installation.
