from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
import launch_ros.parameter_descriptions
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_prefix, get_packages_with_prefixes
import yaml

import os

def generate_launch_description():
    # Define paths for parameter files
    
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Navigate up to the src directory
    workspace_dir = os.path.join('/home/nanli/ros2_ws/', 'src/lmco')
    package_src_dir = os.path.join(workspace_dir, 'ltl_automaton_planner')
    config_dir = os.path.join(package_src_dir, 'config')

    ltl_formula_file = os.path.join(config_dir, 'isaac_ltl_formula.yaml')
    transition_system_file = os.path.join(config_dir, 'isaac_known.yaml')

    ld = LaunchDescription()
    declare_argo_type_cmd = DeclareLaunchArgument(
        'algo_type',
        default_value='relaxed',
        description='Algorithm type (e.g., dstar-relaxed/brute-force/local/relaxed)'
    )
    declare_namespace1_cmd = DeclareLaunchArgument(
        'robot1_namespace',
        default_value='robot1',
        description='Namespace for the first robot'
    )
    declare_namespace2_cmd = DeclareLaunchArgument(
        'robot2_namespace',
        default_value='robot2',
        description='Namespace for the second robot'
    )
    declare_ltl_file_cmd = DeclareLaunchArgument(
        'ltl_params_file',
        default_value=ltl_formula_file,
        description='ltl formula file',
    )  
    declare_ts_file_cmd = DeclareLaunchArgument(
        'ltl_params_file',
        default_value=ltl_formula_file,
        description='ltl formula file',
    )

    
    robot_1_node = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot1_namespace')),
        Node(
            package='ltl_automaton_planner',
            executable='benchmark_node',
            name='benchmark_node',
            output='screen',
            parameters=[ltl_formula_file,
                         {'transition_system_textfile': transition_system_file},
                         {'N': 6}]
        ),
        Node(
            package='ltl_automaton_planner',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[
                ltl_formula_file,
                {'algo_type': LaunchConfiguration('algo_type')},
                {'transition_system_textfile': transition_system_file},
                {'init_state': 'c0_r0'}
            ]
        ),
        # Node(
        #     package='ltl_automaton_planner',
        #     executable='relay_node',
        #     name='relay_node',
        #     output='screen',
        # )
    ])
    robot_2_node = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot2_namespace')),
        Node(
            package='ltl_automaton_planner',
            executable='benchmark_node',
            name='benchmark_node',
            output='screen',
            parameters=[ltl_formula_file,
                         {'transition_system_textfile': transition_system_file},
                         {'N': 6}]
        ),
        Node(
            package='ltl_automaton_planner',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[
                ltl_formula_file,
                {'algo_type': LaunchConfiguration('algo_type')},
                {'transition_system_textfile': transition_system_file},
                {'init_state': 'c4_r3'}
            ]
        ),
        # Node(
        #     package='ltl_automaton_planner',
        #     executable='relay_node',
        #     name='relay_node',
        #     output='screen',
        # )
    ])
    ld.add_action(declare_argo_type_cmd)
    ld.add_action(declare_namespace1_cmd)
    ld.add_action(declare_namespace2_cmd)
    ld.add_action(declare_ltl_file_cmd)
    ld.add_action(declare_ts_file_cmd)
    ld.add_action(robot_1_node)
    ld.add_action(robot_2_node)
    
    return ld

    # return LaunchDescription([
    #     # Declare launch arguments
    #     DeclareLaunchArgument(
    #         'algo_type',
    #         default_value='dstar',
    #         description='Algorithm type (e.g., dstar-relaxed/brute-force/local/relaxed)'
    #     ),
    #     DeclareLaunchArgument(
    #         'robot1_namespace',
    #         default_value='robot1',
    #         description='Namespace for the first robot'
    #     ),
    #     DeclareLaunchArgument(
    #         'robot2_namespace',
    #         default_value='robot2',
    #         description='Namespace for the second robot'
    #     ),

    #     # Group for robot1 namespace
    #     GroupAction([
    #         PushRosNamespace(LaunchConfiguration('robot1_namespace')),
    #         Node(
    #             package='ltl_automaton_planner',
    #             executable='benchmark_node',
    #             name='simulation',
    #             output='screen',
    #             parameters=[ltl_formula_file,
    #                         transition_system_file]
    #         ),
    #         Node(
    #             package='ltl_automaton_planner',
    #             executable='planner_node',
    #             name='ltl_planner',
    #             output='screen',
    #             parameters=[
    #                 ltl_formula_file,
    #                 {'algo_type': LaunchConfiguration('algo_type')},
    #                 {'transition_system_textfile': transition_system_file}
    #             ]
    #         )
    #     ]),

        # Group for robot2 namespace
        # GroupAction([
        #     PushRosNamespace(LaunchConfiguration('robot2_namespace')),
        #     Node(
        #         package='ltl_automaton_planner',
        #         executable='benchmark_node',
        #         name='simulation',
        #         output='screen',
        #         parameters=[ltl_formula_file]
        #     ),
        #     Node(
        #         package='ltl_automaton_planner',
        #         executable='planner_node',
        #         name='ltl_planner',
        #         output='screen',
        #         parameters=[
        #             {'algo_type': LaunchConfiguration('algo_type')},
        #             {'transition_system_textfile': transition_system_file}
        #         ]
        #     )
        # ])
    #])
