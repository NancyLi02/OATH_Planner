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

    ltl_formula_file = os.path.join(config_dir, 'task_ltl.yaml')
    transition_system_file = os.path.join(config_dir, 'isaac_known.yaml')

    ld = LaunchDescription()

    declare_argo_type_cmd = DeclareLaunchArgument(
        'algo_type',
        default_value='brute-force',
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
            parameters=[#ltl_formula_file,
                {'agent_name': 'robot_1'},
                {'transition_system_textfile': transition_system_file},
                {'N': 6}]
        ),
        Node(
            package='ltl_automaton_planner',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[
                {'agent_name': 'robot_1'},
                {'ltl_formula_file': ltl_formula_file},
                {'algo_type': LaunchConfiguration('algo_type')},
                {'transition_system_textfile': transition_system_file},
                {'init_state': 'c0_r0'}
            ]
        ),
    ])

    robot_2_node = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot2_namespace')),
        Node(
            package='ltl_automaton_planner',
            executable='benchmark_node',
            name='benchmark_node',
            output='screen',
            parameters=[#ltl_formula_file,
                {'agent_name': 'robot_2'},
                {'transition_system_textfile': transition_system_file},
                {'N': 6}]
        ),
        Node(
            package='ltl_automaton_planner',
            executable='planner_node',
            name='planner_node',
            output='screen',
            parameters=[
                {'agent_name': 'robot_2'},
                {'ltl_formula_file': ltl_formula_file},
                {'algo_type': LaunchConfiguration('algo_type')},
                {'transition_system_textfile': transition_system_file},
                {'init_state': 'c4_r3'}
            ]
        ),
    ])

    taskassign_node = Node(
        package='ltl_automaton_planner',
        executable='taskassign_node',
        name='taskassign_node',
        output='screen',
        parameters=[
            {'robot1_initial_state': 'c0_r0'},
            {'robot2_initial_state': 'c4_r3'},
            {'score_scheme': 'dstar'} # dstar or manhattan
        ]
    )

    ld.add_action(declare_argo_type_cmd)
    ld.add_action(declare_namespace1_cmd)
    ld.add_action(declare_namespace2_cmd)
    ld.add_action(declare_ltl_file_cmd)
    ld.add_action(declare_ts_file_cmd)
    ld.add_action(robot_1_node)
    ld.add_action(robot_2_node)
    ld.add_action(taskassign_node)

    return ld
