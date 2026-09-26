from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the stage-1 color detector against the Gazebo D435 streams."""
    target_color = LaunchConfiguration('target_color')

    return LaunchDescription([
        DeclareLaunchArgument(
            'target_color',
            default_value='red',
            description="color preset: red / blue / green / yellow / custom"),
        Node(
            package='mycobot_vision_tutorial',
            executable='color_detector',
            name='color_detector',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'target_color': target_color,
            }],
        ),
    ])
