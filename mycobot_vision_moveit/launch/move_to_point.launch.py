import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE_NAME = 'mycobot_moveit_config'
ROBOT_NAME = 'mycobot_280'


def _normalize(obj):
    """Convert tuples to plain lists (yaml-safe)."""
    if isinstance(obj, tuple):
        return [_normalize(x) for x in obj]
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    return obj


def moveit_params():
    """Mirror the MoveItConfigsBuilder chain of move_group.launch.py."""
    config_path = os.path.join(
        get_package_share_directory(PACKAGE_NAME), 'config', ROBOT_NAME)

    moveit_config = (
        MoveItConfigsBuilder(ROBOT_NAME, package_name=PACKAGE_NAME)
        .robot_description_semantic(
            file_path=os.path.join(config_path, f'{ROBOT_NAME}.srdf'))
        .joint_limits(file_path=os.path.join(config_path, 'joint_limits.yaml'))
        .robot_description_kinematics(
            file_path=os.path.join(config_path, 'kinematics.yaml'))
        .planning_pipelines(
            pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )
    return _normalize(moveit_config.to_dict())


def launch_setup(context):
    approach_height = float(LaunchConfiguration('approach_height').perform(context))
    single_shot = LaunchConfiguration('single_shot').perform(context).lower() in ('true', '1')

    params = {
        **moveit_params(),
        'use_sim_time': True,
        'approach_height': approach_height,
        'single_shot': single_shot,
    }
    params_file = os.path.join(
        tempfile.gettempdir(), 'mycobot_vision_moveit_params.yaml')
    with open(params_file, 'w') as fh:
        yaml.safe_dump({'/**': {'ros__parameters': params}}, fh)

    # Hand the params file over via env: the node injects it into rclcpp::init
    # itself. Going through launch's own --ros-args sections is fragile
    # (python/tuple tags, trailing empty sections).
    return [Node(
        package='mycobot_vision_moveit',
        executable='move_to_point',
        output='screen',
        additional_env={'MYCOBOT_VISION_MOVEIT_PARAMS': params_file},
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'approach_height', default_value='0.03',
            description='hover height above the detected point [m]'),
        DeclareLaunchArgument(
            'single_shot', default_value='true',
            description='move only on the first detection'),
        OpaqueFunction(function=launch_setup),
    ])
