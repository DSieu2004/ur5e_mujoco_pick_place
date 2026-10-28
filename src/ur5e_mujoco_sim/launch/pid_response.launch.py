"""Fresh camera-equipped PID simulation and measured small-step experiment."""

from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("ur5e_mujoco_sim"))
    workspace = Path(get_package_prefix("ur5e_mujoco_sim")).parents[1]
    return LaunchDescription([
        DeclareLaunchArgument("vision_python",
                              default_value=str(workspace / ".venv-vision/bin/python")),
        DeclareLaunchArgument("step_rad", default_value="0.05"),
        DeclareLaunchArgument("hold_s", default_value="3.0"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch/vision_pick_place.launch.py")),
            launch_arguments={"run_task": "false", "preview": "false"}.items()),
        TimerAction(period=3.0, actions=[Node(
            package="ur5e_mujoco_sim", executable="pid_step_response", output="screen",
            sigterm_timeout="15",
            prefix=LaunchConfiguration("vision_python"),
            parameters=[{
                "step_rad": ParameterValue(LaunchConfiguration("step_rad"), value_type=float),
                "hold_s": ParameterValue(LaunchConfiguration("hold_s"), value_type=float)}])]),
    ])
