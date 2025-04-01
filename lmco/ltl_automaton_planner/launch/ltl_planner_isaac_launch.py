from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
import os
import numpy as np
import random

random.seed(42)
np.random.seed(42)
def generate_launch_description():
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    workspace_dir = os.path.join('/home/nanli/ros2_ws/', 'src/lmco')
    package_src_dir = os.path.join(workspace_dir, 'ltl_automaton_planner')
    config_dir = os.path.join(package_src_dir, 'config')
    
    ltl_formula_file = os.path.join(config_dir, 'task_ltl.yaml')
    transition_system_file = os.path.join(config_dir, 'isaac_known.yaml')
    
    ld = LaunchDescription()
    
    declare_robot_count_cmd = DeclareLaunchArgument(
        'robot_count',
        default_value='4',
        description='Number of robots to launch'
    )
    declare_task_count_cmd = DeclareLaunchArgument(
        'task_count',
        default_value='18',
        description='Number of tasks to launch'
    )

    declare_argo_type_cmd = DeclareLaunchArgument(
        'algo_type',
        default_value='brute-force',
        description='Algorithm type (e.g., dstar-relaxed/brute-force/local/relaxed)'
    )

    
    ld.add_action(declare_robot_count_cmd)
    ld.add_action(declare_task_count_cmd)
    ld.add_action(declare_argo_type_cmd)
    
    for i in range(1, 5):  # 4 robots
        robot_namespace = f'robot{i}'
        agent_name = f'robot_{i}'
        init_state_arg = f'robot{i}_init_state'
        
        declare_namespace_cmd = DeclareLaunchArgument(
            f'{robot_namespace}_namespace',
            default_value=robot_namespace,
            description=f'Namespace for robot {i}'
        )
        declare_init_state_cmd = DeclareLaunchArgument(
            init_state_arg,
            default_value=str(999 + i),
            description=f'Initial state for robot {i}'
        )
        
        ld.add_action(declare_namespace_cmd)
        ld.add_action(declare_init_state_cmd)
        
        robot_node = GroupAction([
            PushRosNamespace(LaunchConfiguration(f'{robot_namespace}_namespace')),
            Node(
                package='ltl_automaton_planner',
                # executable='benchmark_node',
                # name='benchmark_node',
                executable='benchmark_cluster_node',
                name='benchmark_cluster_node',
                output='screen',
                parameters=[
                    {'agent_name': agent_name},
                    {'transition_system_textfile': transition_system_file},
                    {'N': 20},
                    {'init_state': LaunchConfiguration(init_state_arg)}
                ]
            ),
            Node(
                package='ltl_automaton_planner',
                # executable='planner_node',
                # name='planner_node',
                executable='planner_cluster_node',
                name='planner_cluster_node',
                output='screen',
                parameters=[
                    {'agent_name': agent_name},
                    {'ltl_formula_file': ltl_formula_file},
                    {'algo_type': LaunchConfiguration('algo_type')},
                    {'transition_system_textfile': transition_system_file},
                    {'init_state': LaunchConfiguration(init_state_arg)}
                ]
            ),
        ])
        
        ld.add_action(robot_node)
    
    taskassign_node = Node(
        package='ltl_automaton_planner',
        # executable='taskassign_node',
        # name='taskassign_node',
        executable='taskassign_cluster_node',
        name='taskassign_cluster_node',
        output='screen',
        parameters=[
            {'score_scheme': 'dstar'}
        ] + [
            {f'robot{i}_init_pose': LaunchConfiguration(f'robot{i}_init_state')}
            for i in range(1, 5)
        ]
    )

    showmove_node = Node(
        package='ltl_automaton_planner',
        executable='showmove_node',
        name='showmove_node',
        output='screen',
        parameters=[
                    {'transition_system_textfile': transition_system_file},
                    {'N': 20}
                ]
            )

    
    ld.add_action(taskassign_node)
    ld.add_action(showmove_node)
    
    return ld
