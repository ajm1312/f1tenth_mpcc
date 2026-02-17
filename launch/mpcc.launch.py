from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('mpcc'),
        'config',
        'mpcc_config.yaml'
    )


    mpcc_node = Node(
        package='mpcc',
        executable='mpcc_node',
        name='mpcc_node',
        output='screen', # Optional: see node output in terminal
        parameters=[config] # Optional: load parameters
    )

    return LaunchDescription([
        mpcc_node,
    ])
