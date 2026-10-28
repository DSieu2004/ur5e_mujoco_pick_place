"""Start the PID simulation, wrist RGB-D camera and autonomous YOLO task."""

import os
from pathlib import Path
import subprocess
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("ur5e_mujoco_sim"))
    workspace = Path(get_package_prefix("ur5e_mujoco_sim")).parents[1]
    venv = Path(os.environ.get("VIRTUAL_ENV", str(workspace / ".venv-vision")))
    sys.path.insert(0, str(Path(get_package_prefix("ur5e_mujoco_sim")) / "lib/ur5e_mujoco_sim"))
    from vision_lib.calibration import validate_camera_mount

    def check_camera_configuration(context):
        validate_camera_mount(LaunchConfiguration("scene").perform(context),
                              share / "config/ur5e_vision.urdf.xacro")
        return []
    profile = subprocess.check_output(
        [sys.executable, str(share / "pid_lib/write_profile.py"),
         str(share / "config/ros2_controllers.yaml")], text=True).strip()
    description = {"robot_description": ParameterValue(Command([
        "xacro ", str(share / "config/ur5e_vision.urdf.xacro"),
        " mujoco_model:=", LaunchConfiguration("scene")]), value_type=str)}
    semantic = {"robot_description_semantic": ParameterValue(Command([
        "xacro ", str(share / "config/ur5e_vision.srdf.xacro")]), value_type=str)}
    sim_time = {"use_sim_time": True}
    simulator = Node(
        package="mujoco_ros2_control", executable="ros2_control_node", output="screen",
        parameters=[description, sim_time, profile,
                    str(share / "config/mujoco_vision_plugins.yaml")])
    return LaunchDescription([
        DeclareLaunchArgument("vision_python", default_value=str(venv / "bin/python")),
        DeclareLaunchArgument("scene",
                              default_value=str(share / "models/ur5e_robotiq_vision_scene.xml")),
        DeclareLaunchArgument("run_task", default_value="true"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("weights", default_value=str(share / "models/box_detector.pt")),
        DeclareLaunchArgument("confidence", default_value="0.40"),
        DeclareLaunchArgument("preview",
                              default_value="true" if os.environ.get("DISPLAY") else "false"),
        OpaqueFunction(function=check_camera_configuration),
        RegisterEventHandler(OnProcessExit(target_action=simulator, on_exit=[
            EmitEvent(event=Shutdown(reason="MuJoCo simulator stopped"))])),
        simulator,
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[description, sim_time]),
        Node(package="ur5e_mujoco_sim", executable="ur5e_move_group",
             name="move_group", output="screen",
             parameters=[description, semantic, sim_time, str(share / "config/move_group.yaml")]),
        TimerAction(period=0.1, actions=[Node(
            package="controller_manager", executable="spawner",
            arguments=["joint_state_broadcaster", "ur5e_controller", "gripper_controller",
                       "--controller-manager", "/controller_manager", "--param-file", profile],
            output="screen")]),
        TimerAction(period=3.0, actions=[Node(
            package="ur5e_mujoco_sim", executable="vision_pick_place",
            prefix=LaunchConfiguration("vision_python"), output="screen",
            sigterm_timeout="15",
            condition=IfCondition(LaunchConfiguration("run_task")),
            parameters=[{
                "device": ParameterValue(LaunchConfiguration("device"), value_type=str),
                "weights": ParameterValue(LaunchConfiguration("weights"), value_type=str),
                "confidence": ParameterValue(LaunchConfiguration("confidence"), value_type=float)
            }])]),
        Node(package="ur5e_mujoco_sim", executable="vision_camera_preview",
             prefix=LaunchConfiguration("vision_python"), output="screen",
             condition=IfCondition(LaunchConfiguration("preview"))),
    ])
