#!/usr/bin/env python
import os
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import sys
import yaml
import std_msgs
from copy import deepcopy
#Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TaskFail, AgentFail, AgentFailTask, NoTask, TransitionSystemStateStamped, TransitionSystemState,UpdateValidTasks, WaitingRequest, StopWaiting, PositionRequest, TaskRequestCluster, CurrentPosition, LTLPlan, RelayRequest, RelayResponse, ShowPosition, AddTask, ObstacleUpdate
from ltl_automaton_msgs.srv import TaskReplanningDelete, TaskReplanningModify # TaskReplanningAddRequest, TaskReplanningDeleteRequest, TaskReplanningRelabelRequest
# Import transition system loader
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers, build_graph_halton, check_in_block, check_in_bump, add_block_polygon, add_bump_polygon, update_graph_with_obstacle
# Import coordinate transformer for hardware navigation
from ltl_automaton_planner.coordinate_transform import CoordinateTransformer, create_transformer_from_params
# Import modules for commanding the a1

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
import pygame
from enum import Enum
import cv2
import numpy as np
import time
import csv
import json
from shapely.geometry import Point, LineString, Polygon
from example_interfaces.srv import AddTwoInts
import re
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from interfaces_hmm_sim.msg import Status, ReplanStatus, AgentGoTo
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String

# Import Nav2 action for hardware experiments with TurtleBot3
from nav2_msgs.action import NavigateToPose

#=================================================================
#  Interfaces between LTL planner node and lower level controls
#                       -----------------
# The node is reponsible for outputting current agent state based
# on TS and agent output.
# The node converts TS action to interpretable commands using
# action attributes defined in the TS config file
#=================================================================

USE_ISAAC = False
USE_HARDWARE = False  # Set to True for TurtleBot3 hardware experiments

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 255)
ORANGE = (255, 100, 0)
BLUE = (0, 0, 128)

class EquipmentMode(Enum):
    UNLOADED = (0, 255, 0)
    LOADED = (0, 255, 255)
    WAITTASK = (255, 100, 0)
    NOTASK = (96, 96, 96)
    RESCUE = (255, 0, 0)
    FAIL = (0, 0, 0)

class GridWorld(object):
    def __init__(self, grid_size):
         # Constants
        self.grid_size = grid_size
        self.load_elements()
        
        self.width, self.height = 800, 800
        self.cell_size = self.width // self.grid_size

        home_directory = os.path.expanduser("~")

    
    def load_elements(self):
        package_share = get_package_share_directory('ltl_automaton_planner')
        # with open(parent_dir + '/config/benchmark_block_'+str(self.grid_size)+'.yaml', 'r') as file:
        with open(os.path.join(package_share, 'config', 'isaac_block.yaml'), 'r') as file:
            yaml_data = yaml.safe_load(file)

            if isinstance(yaml_data['blocks'], list):
                print("loading all blocks...................")
                wall = yaml_data['blocks']
                self.wall = dict()
                for w in wall:
                    self.wall[(tuple(w[0]),tuple(w[1]))] = 0 
            else:
                print("The YAML file does not contain a list.")     
        
        print(self.wall)
        # with open(parent_dir + '/config/benchmark_bump_'+str(self.grid_size)+'.yaml', 'r') as file:
        with open(os.path.join(package_share, 'config', 'isaac_bump.yaml'), 'r') as file:
            yaml_data = yaml.safe_load(file)

            if isinstance(yaml_data['bumps']['points'], list):
                print("loading all bumps...................")
                bump = yaml_data['bumps']['points']
                self.bump = dict()
                for b in bump:
                    self.bump[(tuple(b[0]),tuple(b[1]))] = 0 
            else:
                print("The YAML file does not contain a list.")    
        
        self.bump = dict()
        self.block = dict()



class LTLControllerDrone(Node):
    def __init__(self, env):
        super().__init__("benchmark_node")
        print(self.get_node_names_and_namespaces())
        self.world = env
        self.action_list = []
        
        self.prefix_action_list = []
        self.suffix_action_list = []
        self.drone_prefix_sub = self.create_subscription(
            LTLPlan,
            'prefix_plan',
            self.prefix_plan_callback,
            10  # QoS History depth
        )
        self.drone_suffix_sub = self.create_subscription(
            LTLPlan,
            'suffix_plan',
            self.suffix_plan_callback,
            10  # QoS History depth (same as ROS1 queue_size)
        )
        self.relay_sub = self.create_subscription(
            RelayResponse,
            'replanning_response',
            self.relay_callback,
            10)
        
        self.position_request_sub = self.create_subscription(
            PositionRequest,
            'position_request',
            self.get_current_pos,
            10
        )

        self.update_pose_sub = self.create_subscription(
            PositionRequest,
            'update_pose_request',
            self.update_current_pos,
            10
        )

        self.waiting_sub = self.create_subscription(
            WaitingRequest,
            'waiting_request',
            self.waiting_request,
            10
        )

        self.stop_waiting_sub = self.create_subscription(
            StopWaiting,
            'stop_waiting',
            self.stop_waiting,
            10
        )

        self.no_task_sub = self.create_subscription(
            NoTask,
            'no_task',
            self.no_task_callback,
            10
        )

        self.agent_fail_sub = self.create_subscription(
            AgentFail,
            'agent_failure',
            self.agent_fail_callback,
            10
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.task_failed_sub = self.create_subscription(
            TaskFail,
            'task_failure',
            self.task_failed_callback,
            qos
        )

        self.add_task_sub = self.create_subscription(
            AddTask,
            "/add_task",
            self.add_task_callback,
            10
        )
        
        self.obstacle_update_sub = self.create_subscription(
            ObstacleUpdate,
            'obstacle_update',
            self.obstacle_update_callback,
            10
        )
        
        self.relay_pub = self.create_publisher(RelayRequest, 'replanning_request', 10)
        self.current_position_pub = self.create_publisher(CurrentPosition,'current_position', 10)
        self.update_pose_pub = self.create_publisher(CurrentPosition,'update_current_pose', 10)
        self.taskassignment_request_pub = self.create_publisher(TaskRequestCluster, 'task_assignment_request', 10)
        self.position_pub = self.create_publisher(ShowPosition, 'show_position', 10)
        self.update_valid_tasks_pub = self.create_publisher(UpdateValidTasks, 'update_valid_tasks', 10)
        self.agent_failed_task_pub = self.create_publisher(AgentFailTask, 'agent_fail_task', 10)
        self.task_failed_task_new_cluster_pub = self.create_publisher(TaskFail, 'task_failure_cluster', 10)

        self.next_issac_step_pub = self.create_publisher(AgentGoTo, 'agent_next', 10)

        self.pub_assign = True
        self.on_hold = False
        self.final_on_hold = False
        self.no_task = False
        self.cur_task = ''
        self.act = ''
        
        transition_system_textfile = self.declare_parameter('transition_system_textfile', '').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.declare_parameter('agent_name', '')
        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        self.declare_parameter('init_state', 0)
        self.init_pose = self.get_parameter('init_state').value

        # ========== Hardware (TurtleBot3) Navigation Setup ==========
        if USE_HARDWARE:
            self.callback_group = ReentrantCallbackGroup()
            # Create Nav2 NavigateToPose action client
            # The action server is namespaced: /<robot_namespace>/navigate_to_pose
            self.nav2_action_client = ActionClient(
                self,
                NavigateToPose,
                'navigate_to_pose',  # Will be namespaced by the robot's namespace
                callback_group=self.callback_group
            )
            self.get_logger().info(f'[{self.agent_name}] Waiting for Nav2 NavigateToPose action server...')
            # Wait for the action server to be available (with timeout)
            if not self.nav2_action_client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error(f'[{self.agent_name}] Nav2 action server not available!')
            else:
                self.get_logger().info(f'[{self.agent_name}] Nav2 action server connected!')
            
            # Hardware navigation state variables
            self.hardware_goal_in_progress = False
            self.hardware_goal_reached = False
            self.hardware_goal_failed = False
            self.hardware_navigation_start = True  # Flag to allow first movement
            self.hardware_goal_pending = False  # Flag to track pending goal when Nav2 not ready
            self.pending_goal_pose = None  # Store the pending goal position
            self._goal_handle = None
            
            # ========== Coordinate Transformation Setup ==========
            # Parameters for transforming pygame map coordinates to real-world meters
            # Default: pygame map 18x18 units -> real map 9ft x 9ft
            self.declare_parameter('pygame_map_width', 18.0)
            self.declare_parameter('pygame_map_height', 18.0)
            self.declare_parameter('real_map_width_ft', 9.0)
            self.declare_parameter('real_map_height_ft', 9.0)
            self.declare_parameter('pygame_origin_x', 0.0)
            self.declare_parameter('pygame_origin_y', 0.0)
            self.declare_parameter('real_origin_x', 0.0)  # Real world origin offset in meters
            self.declare_parameter('real_origin_y', 0.0)  # Real world origin offset in meters
            
            # Get coordinate transformation parameters
            pygame_map_width = self.get_parameter('pygame_map_width').get_parameter_value().double_value
            pygame_map_height = self.get_parameter('pygame_map_height').get_parameter_value().double_value
            real_map_width_ft = self.get_parameter('real_map_width_ft').get_parameter_value().double_value
            real_map_height_ft = self.get_parameter('real_map_height_ft').get_parameter_value().double_value
            pygame_origin_x = self.get_parameter('pygame_origin_x').get_parameter_value().double_value
            pygame_origin_y = self.get_parameter('pygame_origin_y').get_parameter_value().double_value
            real_origin_x = self.get_parameter('real_origin_x').get_parameter_value().double_value
            real_origin_y = self.get_parameter('real_origin_y').get_parameter_value().double_value
            
            # Create coordinate transformer
            self.coord_transformer = create_transformer_from_params(
                pygame_width=pygame_map_width,
                pygame_height=pygame_map_height,
                real_width_ft=real_map_width_ft,
                real_height_ft=real_map_height_ft,
                pygame_origin_x=pygame_origin_x,
                pygame_origin_y=pygame_origin_y,
                real_origin_x=real_origin_x,
                real_origin_y=real_origin_y
            )
            
            self.get_logger().info(f'[{self.agent_name}] Coordinate transformer initialized:')
            self.get_logger().info(f'  Pygame map size: {pygame_map_width} x {pygame_map_height} units')
            self.get_logger().info(f'  Real map size: {real_map_width_ft} x {real_map_height_ft} ft = '
                                   f'{self.coord_transformer.real_map_size_m[0]:.4f} x {self.coord_transformer.real_map_size_m[1]:.4f} m')
            self.get_logger().info(f'  Scale factors: {self.coord_transformer.scale_x:.6f} m/unit (x), {self.coord_transformer.scale_y:.6f} m/unit (y)')
            self.get_logger().info(f'  Real world origin offset: ({real_origin_x}, {real_origin_y}) m')

        self.nodes, generated_actions = build_graph_halton(18, 18, 100)
        
        self.transition_system ['state_models']['2d_pose_region']['nodes'] = self.nodes
        self.transition_system ['actions'].update(generated_actions)
        # IMPORTANT: self.actions must point to the same object as transition_system['actions']
        self.actions = self.transition_system['actions']

        self.mode = EquipmentMode.UNLOADED
        self.total_cost = 0
        self.if_obs = False


        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        robot_positions = yaml_data.get('robot_positions', {})
        if self.agent_name in robot_positions:
            robot_pos_str = robot_positions[self.agent_name]
            coords = robot_pos_str.split(',')
            if len(coords) == 2:
                self.pose = (float(coords[0]), float(coords[1]))
            else:
                self.get_logger().warn(f'Invalid robot position format for {self.agent_name}: {robot_pos_str}')
                self.pose = (0, 0)
        else:
            self.get_logger().warn(f'Robot {self.agent_name} not found in Task_Points.yaml')
            self.pose = (0, 0)

        self.pose_index = self.init_pose

        self.previous_pose = self.pose
        self.previous_pose_index = self.pose_index
        self.pose_history = [(self.pose, 0)]
        self.t_sim = self.get_clock().now()  # Use the ROS2 clock for the current time
        self.plan_index = 0
        self.total_plan_index = 0
        self.next_interval = 10

        self.create_timer(1.0/10, self.simulate)

        self.status_sub = self.create_subscription(
            Status,
            'status',
            self.status_callback,
            10)
        
        self.replan_pub = self.create_publisher(
            ReplanStatus,
            'replan_status',
            10
        )

        self.sim_arrived = True
        self.sim_received = True
        self.sim_start = True

        self.new_task_points = []
        self.processed_obstacles = set()
        
        # Step counter for timing summary
        self.total_steps = 0
        
        # Timing data publisher
        self.timing_pub = self.create_publisher(String, '/timing_data', 10)
        
        # Subscribe to global completion message (all robots finished)
        self.all_robots_finished_sub = self.create_subscription(
            String,
            '/all_robots_finished',
            self.all_robots_finished_callback,
            10
        )


        def str_to_tuple(s):
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            return tuple(float(x) if '.' in x else int(x) for x in nums)
        package_share = get_package_share_directory('ltl_automaton_planner')
        task_points_yaml = os.path.join(package_share, 'config', 'Task_Points.yaml')
        with open(task_points_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        self.task_points = {str_to_tuple(k): v for k, v in yaml_data['task_points'].items()}
        self.task_to_delivery = yaml_data['task_to_delivery']

        # self.simulate()

    def obstacle_update_callback(self, msg):
        # Record response start time (monotonic for accurate duration)
        response_start_mono = time.monotonic()
        response_start_time = time.time()
        
        obstacle_key = tuple(msg.obstacle_location)
        if obstacle_key in self.processed_obstacles:
            self.get_logger().info(f"Ignoring duplicate obstacle update for location: {msg.obstacle_location}")
            return
            
        self.get_logger().info(f"Received obstacle update: {msg.obstacle_type} at {msg.obstacle_location}")
        self.processed_obstacles.add(obstacle_key)
        # Don't clear world.block and world.bump here - let them accumulate for visualization

        # Convert flattened coordinates back to list of tuples
        coords_flat = msg.obstacle_location
        if len(coords_flat) % 2 != 0:
            self.get_logger().error("Received obstacle with an odd number of coordinates.")
            return
        
        vertices = []
        for i in range(0, len(coords_flat), 2):
            vertices.append((coords_flat[i], coords_flat[i+1]))

        if msg.obstacle_type == 'wall' or msg.obstacle_type == 'block':
            new_obstacle = Polygon(vertices)
            # Add obstacle to BLOCK_POLYGONS so check_in_block() in next_move() can detect it
            # This ensures the robot won't walk through even if replan fails
            add_block_polygon(vertices)
            # Also store the obstacle for the current path check
            self.get_logger().info("New wall obstacle received. Added to BLOCK_POLYGONS. Checking current path for collision...")
            self.check_path_for_new_obstacle(new_obstacle)
            
            # Record response end time and publish timing (monotonic for accurate duration)
            response_end_mono = time.monotonic()
            response_duration = response_end_mono - response_start_mono
            self.get_logger().info(f'[TIMING] [{self.agent_name}] obstacle_update response completed (including path check). Duration: {response_duration:.4f}s')
            
            timing_msg = String()
            timing_msg.data = json.dumps({
                'type': 'system_response',
                'command': 'obstacle_update',
                'node': 'benchmark_cluster_node',
                'agent_name': self.agent_name,
                'obstacle_type': msg.obstacle_type,
                'start_time': response_start_time,
                'end_time': time.time(),
                'duration': response_duration,
                'timestamp': time.time()
            })
            self.timing_pub.publish(timing_msg)
        elif msg.obstacle_type == 'bush' or msg.obstacle_type == 'bump':
            add_bump_polygon(vertices)
            self.get_logger().info("New bump obstacle added.")
            # Bumps add cost, but don't block, so we might not need to replan immediately unless the path becomes too costly
            # Or we can just let the existing bump logic handle it on the next move.
        
    def check_path_for_new_obstacle(self, new_obstacle_polygon):
        """
        Check if current path passes through the new obstacle.
        If YES: request local replanning via RelayRequest (similar to block detection in next_move)
        If NO: keep current plan and continue
        
        The graph update will happen in the NEXT task round.
        """
        # We need to check if the current or upcoming path segment is now blocked.
        if len(self.prefix_action_list) == 0 and len(self.suffix_action_list) == 0:
            self.get_logger().info("No active plan. New obstacle will be applied in next task round.")
            return

        # Check prefix plan
        affected_action = None
        affected_index = -1
        
        for i in range(self.plan_index, len(self.prefix_action_list)):
            action_to_check = self.prefix_action_list[i]

            if not str(action_to_check).startswith("from_"):
                continue

            # Check if this action's path intersects with the new obstacle
            from_idx, to_idx = extract_numbers(str(action_to_check))
            if str(from_idx) not in self.nodes or str(to_idx) not in self.nodes:
                continue
                
            from_pose = self.nodes[str(from_idx)]['attr']['pose']
            to_pose = self.nodes[str(to_idx)]['attr']['pose']
            
            # Create a line segment for this action
            action_line = LineString([from_pose, to_pose])
            
            # Check if this line intersects the new obstacle
            if action_line.intersects(new_obstacle_polygon):
                self.get_logger().warn(f"Path affected: Action '{action_to_check}' intersects with new obstacle.")
                affected_action = action_to_check
                affected_index = i
                break
        
        # Also check suffix plan if prefix is not affected
        if affected_action is None and len(self.suffix_action_list) > 0:
            suffix_start = max(0, self.plan_index - len(self.prefix_action_list))
            for i in range(suffix_start, len(self.suffix_action_list)):
                action_to_check = self.suffix_action_list[i]
                
                if not str(action_to_check).startswith("from_") and not str(action_to_check).startswith("goto"):
                    continue
                
                from_idx, to_idx = extract_numbers(str(action_to_check))
                if from_idx is None or to_idx is None:
                    continue
                if str(from_idx) not in self.nodes or str(to_idx) not in self.nodes:
                    continue
                    
                from_pose = self.nodes[str(from_idx)]['attr']['pose']
                to_pose = self.nodes[str(to_idx)]['attr']['pose']
                
                action_line = LineString([from_pose, to_pose])
                
                if action_line.intersects(new_obstacle_polygon):
                    self.get_logger().warn(f"Suffix path affected: Action '{action_to_check}' intersects with new obstacle.")
                    affected_action = action_to_check
                    affected_index = len(self.prefix_action_list) + i
                    break
        
        if affected_action is None:
            self.get_logger().info("Current path is NOT affected by new obstacle. Continuing with current plan.")
            self.get_logger().info("New obstacle will be applied in next task round.")
            return
        
        # Path IS affected - request local replanning via RelayRequest
        self.get_logger().info(f"Current path IS affected by new obstacle at action index {affected_index}. Requesting local replan...")
        self.get_logger().info(f"Current robot position: pose={self.pose}, pose_index={self.pose_index}, plan_index={self.plan_index}")
        
        # Get the affected edge's from/to poses
        from_idx, to_idx = extract_numbers(str(affected_action))
        from_pose = self.nodes[str(from_idx)]['attr']['pose']
        to_pose = self.nodes[str(to_idx)]['attr']['pose']
        
        # Determine the current state for replanning - use self.plan_index (current robot position)
        # NOT affected_index (where the blocked action is)
        if self.plan_index < len(self.prefix_state_sequence):
            current_state = self.prefix_state_sequence[self.plan_index]
        elif len(self.suffix_state_sequence) > 0:
            suffix_idx = (self.plan_index - len(self.prefix_action_list)) % len(self.suffix_state_sequence)
            current_state = self.suffix_state_sequence[suffix_idx]
        else:
            # Fallback: use the last available state
            current_state = self.prefix_state_sequence[-1] if self.prefix_state_sequence else None
            if current_state is None:
                self.get_logger().error("No valid state for replanning!")
                return
        
        # IMPORTANT: Save current pose before requesting replan
        # This ensures relay_callback resets to the correct position (not some old position)
        self.previous_pose = self.pose
        self.previous_pose_index = self.pose_index
        
        # Send replanning request (type "delete" to remove the blocked edge)
        # exec_index should be self.plan_index (current robot position), NOT affected_index
        try: 
            self.on_hold = True
            publish_msg = RelayRequest()
            publish_msg.type = "delete"
            publish_msg.current_state = current_state
            publish_msg.from_pose.extend(list(from_pose))
            publish_msg.to_pose.extend(list(to_pose))
            publish_msg.exec_index = self.plan_index  # Use current position, not affected action index
            publish_msg.cost = 0.0
            self.relay_pub.publish(publish_msg)
            self.get_logger().info(f"Published local replan request for obstacle-blocked edge from {from_pose} to {to_pose}, exec_index={self.plan_index}")
        except Exception as e:
            self.get_logger().error(f'Failed to publish replan request: {e}')


    def add_task_callback(self, msg):
        self.get_logger().info(f"Received add task command, new {msg.task_type} task appears at {msg.location}, updating map for benchmark......")
        point = tuple(msg.location)
        label = msg.task_label
        # Add a check for duplicates
        new_task = (point, label)
        if new_task not in self.new_task_points:
            self.new_task_points.append(new_task)
            self.get_logger().info(f"New task point {point} with label {label} will be added to the map on next plan.")
        else:
            self.get_logger().info(f"Task point {point} with label {label} already scheduled for addition.")

    def agent_fail_callback(self, msg):
        self.get_logger().info(f'{self.agent_name} is fail, need find new agent compeleting remaining tasks....')
        self.final_on_hold = True
        self.mode = EquipmentMode.FAIL
        agent_fail_task = AgentFailTask()
        agent_fail_task.robot_id = int(re.findall(r'\d+', self.agent_name)[0])
        agent_fail_task.task_id = self.cur_task
        self.agent_failed_task_pub.publish(agent_fail_task)

    def task_failed_callback(self, msg):
        self.get_logger().info(f'{self.agent_name} fail to complete task{self.cur_task}, finding new agents finish this task!!!!!!!!!!!!!!!!!')
        task_fail_msg = TaskFail()
        task_fail_msg.agent_id = msg.agent_id
        task_fail_msg.task_label = msg.task_label
        self.task_failed_task_new_cluster_pub.publish(task_fail_msg)
        self.on_hold = True
        self.publish_task_request()
        self.mode = EquipmentMode.UNLOADED

    def waiting_request(self, msg):
        self.on_hold = True

    def stop_waiting(self, msg):
        self.on_hold = False

    def no_task_callback(self, msg):
        self.no_task = True
        self.final_on_hold = True
        self.mode = EquipmentMode.NOTASK
        
        # Log that this robot is now idle, but don't publish final timing yet
        # Wait for all robots to be in no_task state (global completion)
        self.get_logger().info(f'[{self.agent_name}] Entered no_task state. Current steps: {self.total_steps}')
        self.get_logger().info(f'[{self.agent_name}] Waiting for all robots to finish or new task assignment...')

    def all_robots_finished_callback(self, msg):
        """
        Called when all robots are in no_task state.
        Publish final timing data for this robot.
        """
        self.get_logger().info(f'[{self.agent_name}] Received ALL_ROBOTS_FINISHED signal!')
        self.get_logger().info(f'[TIMING] {self.agent_name} FINAL stats - Total steps: {self.total_steps}, Total plan index: {self.total_plan_index}')
        
        # Publish final timing data for this robot
        timing_msg = String()
        timing_msg.data = json.dumps({
            'type': 'robot_finished',
            'agent_name': self.agent_name,
            'total_steps': self.total_steps,
            'total_plan_index': self.total_plan_index,
            'final': True,
            'timestamp': time.time()
        })
        self.timing_pub.publish(timing_msg)
        

    
    def get_current_pos(self, msg=None):
        self.get_logger().info('Getting Current Position Now ................................')
        current_pos_msg = CurrentPosition()
        current_pos_msg.robot_id = self.agent_name
        current_pos_msg.pose_index = self.pose_index
        if self.mode == EquipmentMode.LOADED:
            current_pos_msg.current_state = 'loaded'
        else:
            current_pos_msg.current_state = 'unloaded'
            self.on_hold = True
        self.current_position_pub.publish(current_pos_msg)
        self.get_logger().info(f"Publishing current position for {self.agent_name}: {self.pose_index}, with state {current_pos_msg.current_state}.")

    def update_current_pos(self, msg=None):
        self.get_logger().info('Updating Current Position Now ................................')
        current_pos_msg = CurrentPosition()
        current_pos_msg.robot_id = self.agent_name
        current_pos_msg.pose_index = self.pose_index
        if self.mode == EquipmentMode.LOADED:
            current_pos_msg.current_state = 'loaded'
        else:
            current_pos_msg.current_state = 'unloaded'
        self.update_pose_pub.publish(current_pos_msg)
        self.get_logger().info(f"Publishing update current pose index for {self.agent_name}: {self.pose_index}, with state {current_pos_msg.current_state}.")

    def prefix_plan_callback(self, msg):
        if self.new_task_points:
            self.get_logger().info("New task points detected. Rebuilding transition system and updating task points...")
            # Update the task_points dictionary
            for point, label in self.new_task_points:
                self.task_points[point] = label
            # Rebuild the transition system with the new points
            self.nodes, generated_actions = build_graph_halton(20, 20, 1000, self.new_task_points)
            self.transition_system['state_models']['2d_pose_region']['nodes'] = self.nodes
            # Clear old movement actions and add new ones
            keys_to_delete = [k for k in self.transition_system['actions'] if k.startswith('from_')]
            for k in keys_to_delete:
                del self.transition_system['actions'][k]
            self.transition_system['actions'].update(generated_actions)
            self.actions = self.transition_system['actions']
            self.get_logger().info("Transition system and task points updated.")
            # Clear the list after updating
            self.new_task_points.clear()

        # Reset no_task and final_on_hold flags when receiving new task
        # This allows robots to recover from no_task state
        if self.no_task or self.final_on_hold:
            self.get_logger().info(f"[{self.agent_name}] Recovering from no_task state - new task received!")
            self.no_task = False
            self.final_on_hold = False

        # Reset hardware navigation state when receiving new task
        if USE_HARDWARE:
            self.hardware_goal_in_progress = False
            self.hardware_goal_reached = False
            self.hardware_goal_failed = False
            self.hardware_navigation_start = True  # Allow first movement
            self.hardware_goal_pending = False  # Clear any pending goals
            self.pending_goal_pose = None
            if self._goal_handle is not None:
                self.cancel_nav2_goal()  # Cancel any ongoing navigation
            self.get_logger().info(f"[{self.agent_name}] Hardware navigation state reset for new task.")

        self.plan_index = 0
        self.cur_task_list = msg.route_labels
        self.i = 0
        self.cur_task = self.cur_task_list[self.i]
        self.world.block.clear()
        self.world.bump.clear()
        
        # Always reset to UNLOADED when receiving new task
        self.mode = EquipmentMode.UNLOADED
        self.on_hold = False
        # self.get_logger().info("receive data pre")
        self.prefix_action_list = msg.action_sequence
        # self.get_logger().info(f"length prefix_action_list: {len(self.prefix_action_list)}")
        self.prefix_state_sequence = msg.ts_state_sequence
        # self.get_logger().info("end data pre")
        # self.get_logger().info(f'Prefix list received is {self.prefix_action_list}')
        
        # self.prefix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]

    def suffix_plan_callback(self, msg):
        # self.on_hold = False
        # self.get_logger().info("receive data sub")
        self.suffix_action_list = msg.action_sequence
        # self.get_logger().info(f"length suffix_action_list: {len(self.suffix_action_list)}")
        self.suffix_state_sequence = msg.ts_state_sequence
        # self.get_logger().info("end data sub")
        # self.get_logger().info(f'Suffix list received is {self.suffix_action_list}')
        
        # self.suffix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]
        
    def relay_callback(self, msg):
        self.get_logger().info(f"[{self.agent_name}] Received replanning response. Success: {msg.success}")
        self.pose = self.previous_pose
        self.pose_index = self.previous_pose_index
        
        # Cancel any ongoing Nav2 navigation when replanning
        if USE_HARDWARE and self._goal_handle is not None:
            self.cancel_nav2_goal()
            self.hardware_goal_in_progress = False
        
        if msg.success:
            self.prefix_action_list = msg.new_plan_prefix.action_sequence
            self.prefix_state_sequence = msg.new_plan_prefix.ts_state_sequence
            self.suffix_action_list = msg.new_plan_suffix.action_sequence
            self.suffix_state_sequence = msg.new_plan_suffix.ts_state_sequence
            # NOTE: Do NOT reset plan_index! The replanning returns a modified plan
            # where actions before plan_index stay the same, only future actions change.
            self.on_hold = False
            
            # Allow hardware navigation to continue after successful replan
            if USE_HARDWARE:
                self.hardware_navigation_start = True
                self.hardware_goal_reached = False
                self.hardware_goal_failed = False
            
            self.get_logger().info(f"[{self.agent_name}] Local replanning succeeded. Resuming with new plan at index {self.plan_index}.")
        else:
            # Local replanning failed - no alternative path found
            # Fall back to requesting a new task assignment
            self.get_logger().warn(f"[{self.agent_name}] Local replanning FAILED. Requesting new task assignment as fallback.")
            self.prefix_action_list = []
            self.suffix_action_list = []
            self.on_hold = True
            self.publish_task_request()

    # ========== Hardware (TurtleBot3) Navigation Methods ==========
    def send_nav2_goal(self, target_x, target_y, target_yaw=0.0):
        """
        Send a navigation goal to Nav2 for TurtleBot3.
        
        The input coordinates are in pygame map units. They will be automatically
        transformed to real-world meters using the coordinate transformer before
        being sent to Nav2.
        
        Args:
            target_x: Target x coordinate in pygame map units
            target_y: Target y coordinate in pygame map units
            target_yaw: Target orientation (yaw) in radians (default: 0.0)
        """
        if not USE_HARDWARE:
            return
        
        # Transform pygame coordinates to real-world meters
        pygame_coord = (target_x, target_y)
        real_coord = self.coord_transformer.pygame_to_real(pygame_coord)
        real_x, real_y = real_coord
        
        # Always log the goal point for debugging (even if Nav2 not available)
        self.get_logger().info(f'[{self.agent_name}] [DEBUG] Pygame coord: ({target_x:.4f}, {target_y:.4f}) -> '
                               f'Real coord: ({real_x:.4f}m, {real_y:.4f}m), yaw={target_yaw:.4f}')
        
        # Check if action server is available before sending
        if not self.nav2_action_client.server_is_ready():
            self.get_logger().warn(f'[{self.agent_name}] Nav2 action server not ready! Waiting for connection...')
            self.get_logger().warn(f'[{self.agent_name}] Target goal (pygame): ({target_x:.4f}, {target_y:.4f})')
            self.get_logger().warn(f'[{self.agent_name}] Target goal (real): ({real_x:.4f}m, {real_y:.4f}m)')
            # Wait for Nav2 to be ready - mark goal as pending for retry
            self.hardware_goal_in_progress = False
            self.hardware_goal_reached = False  # Keep waiting, don't proceed
            self.hardware_goal_failed = False
            self.hardware_goal_pending = True  # Mark that we have a pending goal
            self.pending_goal_pose = (target_x, target_y)  # Store the PYGAME goal for retry
            return
            
        # Create the goal message with REAL-WORLD coordinates (in meters)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(real_x)
        goal_msg.pose.pose.position.y = float(real_y)
        goal_msg.pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion (only rotation around z-axis)
        import math
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(target_yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(target_yaw / 2.0)
        
        self.hardware_goal_in_progress = True
        self.hardware_goal_reached = False
        self.hardware_goal_failed = False
        self.hardware_goal_pending = False  # Clear pending flag - goal is being sent
        self.pending_goal_pose = None
        
        self.get_logger().info(f'[{self.agent_name}] Sending Nav2 goal: pygame({target_x:.2f}, {target_y:.2f}) -> real({real_x:.4f}m, {real_y:.4f}m)')
        
        # Send the goal asynchronously
        send_goal_future = self.nav2_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav2_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav2_goal_response_callback)
    
    def nav2_goal_response_callback(self, future):
        """Callback when Nav2 accepts or rejects the goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'[{self.agent_name}] Nav2 goal was rejected!')
            self.hardware_goal_in_progress = False
            self.hardware_goal_failed = True
            return
        
        self.get_logger().info(f'[{self.agent_name}] Nav2 goal accepted, waiting for result...')
        self._goal_handle = goal_handle
        
        # Request the result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)
    
    def nav2_result_callback(self, future):
        """Callback when Nav2 navigation completes."""
        result = future.result().result
        status = future.result().status
        
        # ActionGoalStatus: SUCCEEDED=4, CANCELED=5, ABORTED=6
        from action_msgs.msg import GoalStatus
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'[{self.agent_name}] Nav2 goal SUCCEEDED! Robot reached the target.')
            self.hardware_goal_reached = True
            self.hardware_goal_failed = False
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn(f'[{self.agent_name}] Nav2 goal was CANCELED.')
            self.hardware_goal_reached = False
            self.hardware_goal_failed = True
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(f'[{self.agent_name}] Nav2 goal ABORTED! Error: {result.error_msg if hasattr(result, "error_msg") else "unknown"}')
            self.hardware_goal_reached = False
            self.hardware_goal_failed = True
        else:
            self.get_logger().warn(f'[{self.agent_name}] Nav2 goal finished with status: {status}')
            self.hardware_goal_reached = False
            self.hardware_goal_failed = True
        
        self.hardware_goal_in_progress = False
        self._goal_handle = None
    
    def nav2_feedback_callback(self, feedback_msg):
        """Callback for Nav2 navigation feedback (current progress)."""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose
        distance_remaining = feedback.distance_remaining
        # Uncomment for verbose feedback logging:
        # self.get_logger().info(f'[{self.agent_name}] Nav2 feedback - Distance remaining: {distance_remaining:.2f}m')
    
    def cancel_nav2_goal(self):
        """Cancel the current Nav2 navigation goal."""
        if not USE_HARDWARE or self._goal_handle is None:
            return
        
        self.get_logger().info(f'[{self.agent_name}] Canceling Nav2 goal...')
        cancel_future = self._goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.nav2_cancel_callback)
    
    def nav2_cancel_callback(self, future):
        """Callback when Nav2 goal cancellation completes."""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info(f'[{self.agent_name}] Nav2 goal successfully canceled.')
        else:
            self.get_logger().warn(f'[{self.agent_name}] Nav2 goal cancellation failed.')

    def status_callback(self, msg):
        self.sim_arrived = msg.arrived
        self.sim_received = msg.replan_received
        self.sim_start = msg.start
        if (self.sim_arrived):
            self.get_logger().info("sim action complete")
        elif (self.sim_received):
            self.get_logger().info("sim replan received")
        # elif (self.sim_start):
        #     self.get_logger().info("sim started")

    def next_move(self):
        # if self.plan_index > 20 and self.pose == (grid_size/2-1, grid_size/2-1): #self.len(self.prefix_action_list) + len(self.suffix_action_list):
        #     sys.exit()
        #--------------
        # Move command
        #--------------
        # self.get_logger().info("inside the next move")
        if (len(self.prefix_action_list) + len(self.suffix_action_list) != 0):
            # self.get_logger().info(f"plan index: {self.plan_index}")
            if self.plan_index < len(self.prefix_action_list):
                self.pub_assign = True
                #self.get_logger().info("beanchmark fix 0")
                for act in self.transition_system['actions']:
                    #self.get_logger().info("beanchmark fix 0.1")
                    if str(act) == self.prefix_action_list[self.plan_index]:
                        # self.get_logger().info(str(act))
                        #self.get_logger().info("beanchmark fix 0.2")
                        # Extract action types, attributes, etc. in dictionary
                        action_dict = self.transition_system['actions'][str(act)]
                        # print(self.prefix_action_list)
                        if str(act)[:4] == "from":
                            # self.get_logger().info("beanchmark fix 0.3")
                            self.previous_pose = self.pose
                            self.previous_pose_index = self.pose_index
                            self.pose_index = extract_numbers(str(act))[1]
                            self.pose = self.nodes[f'{self.pose_index}']['attr']['pose']
                            self.act = 'g'
                            for pt, label in self.task_points.items():
                                if abs(self.pose[0] - pt[0]) < 1e-6 and abs(self.pose[1] - pt[1]) < 1e-6:
                                    # Only publish UpdateValidTasks if this task is in the assigned task list
                                    if hasattr(self, 'cur_task_list') and label in self.cur_task_list:
                                        self.mode = EquipmentMode.LOADED
                                        msg = UpdateValidTasks()
                                        msg.robot_id = int(re.findall(r'\d+', self.agent_name)[0])
                                        msg.loaded_task = label
                                        self.update_valid_tasks_pub.publish(msg)
                                        self.act = 'l'
                                        # self.get_logger().info(f'Published UpdateValidTasks: robot_id={self.agent_name}, loaded_task={msg.loaded_task}')
                                    break
                            # self.get_logger().info(f"previous pose: {self.previous_pose}")
                            # self.get_logger().info(f"pose: {self.pose}")
                            
                            if check_in_block(act, self.nodes) and str(act) not in self.world.block:
                                # self.get_logger().info("--------Block detected---------")
                                self.world.block[str(act)] = 1
                                # self.if_obs = True
                                # Cancel any ongoing Nav2 navigation when obstacle detected
                                if USE_HARDWARE and hasattr(self, '_goal_handle') and self._goal_handle is not None:
                                    self.cancel_nav2_goal()
                                    self.hardware_goal_in_progress = False
                                try: 
                                    self.on_hold = True
                                    publish_msg = RelayRequest()
                                    # self.get_logger().info("checkpoint2")
                                    publish_msg.type = "delete"
                                    publish_msg.current_state = self.prefix_state_sequence[self.plan_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = 0.0
                                    # self.get_logger().info("checkpoint3")
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    self.pose_index = self.previous_pose_index
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                            # self.get_logger().info("beanchmark fix 0.4")       
                            if check_in_bump(act, self.nodes, self.agent_name) and str(act) not in self.world.bump:
                                # self.get_logger().info("--------Bump detected---------")
                                self.world.bump[str(act)] = 1
                                # self.if_obs = True
                                try: 
                                    publish_msg = RelayRequest()
                                    publish_msg.type = "modify"
                                    publish_msg.current_state = self.prefix_state_sequence[self.plan_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = self.actions[act]['weight']*5
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                            # self.get_logger().info("beanchmark fix 0.5")    
                            # self.pose = (int(str(act).split('c')[1]), int(str(act).split('r')[1]))
                        elif str(act) == "unload" or str(act) == "release":
                            self.mode = EquipmentMode.UNLOADED
                            self.act = 'u'
                        elif str(act) == "load":
                            self.mode = EquipmentMode.LOADED
                            self.act = 'l'
                            # 只有到达任务点时才发布UpdateValidTasks
                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
                        self.total_plan_index += 1
                        self.total_steps += 1  # Increment step counter
                        # self.get_logger().info(f"plan index: {self.plan_index}")
                        print(self.mode)
                        self.t = self.get_clock().now().to_msg()
                        self.next_interval = action_dict['weight']*5 # +1
                        # self.get_logger().info("beanchmark fix 0.6")
                        
                        ##### Logging
                        last_round_time = 50 if (self.previous_pose, self.pose) in self.world.bump else 10
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)-1):
                            pose_ab = extract_numbers(str(act))
                            
                            # pose_a = extract_numbers(str(self.prefix_state_sequence[i].states[0]))
                            # pose_b = extract_numbers(str(self.prefix_state_sequence[i+1].states[0]))
                            if pose_ab not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                future_time += 10
                            else:
                                future_time += 50
                        for i in range(0, len(self.suffix_state_sequence)):
                            pose_ab = extract_numbers(str(act))
                            # pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                            # pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                            if pose_ab not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                future_time += 10
                            else:
                                future_time += 50
                        self.pose_history.append((self.pose, round(last_round_time+last_time), round(future_step), round(future_time)))
                        # self.get_logger().info("beanchmark fix 0.7")
                        return

            if self.plan_index >= len(self.prefix_action_list) and self.plan_index < len(self.prefix_action_list) + len(self.suffix_action_list):
                suffix_index = (self.plan_index - len(self.prefix_action_list)) % len(self.suffix_action_list)
                self.pub_assign = True
                for act in self.transition_system['actions']:
                    if str(act) == self.suffix_action_list[suffix_index]:
                        # Extract action types, attributes, etc. in dictionary
                        action_dict = self.transition_system['actions'][str(act)]
                        print(self.plan_index)
                        # print(act)
                        # print(self.suffix_action_list)
                        if str(act)[:4] == "goto":
                            self.previous_pose = self.pose
                            self.previous_pose_index = self.pose_index
                            self.pose_index = extract_numbers(str(act))[1]
                            self.pose = self.nodes[f'{self.pose_index}']['attr']['pose']
                            self.act = 'g'
                            if check_in_block(act, self.nodes) and str(act) not in self.world.block:
                                # self.get_logger().info("--------Block detected---------")
                                self.world.block[str(act)] = 1
                                # self.if_obs = True
                                # Cancel any ongoing Nav2 navigation when obstacle detected
                                if USE_HARDWARE and hasattr(self, '_goal_handle') and self._goal_handle is not None:
                                    self.cancel_nav2_goal()
                                    self.hardware_goal_in_progress = False
                                try: 
                                    self.on_hold = True
                                    publish_msg = RelayRequest()
                                    # self.get_logger().info("checkpoint2")
                                    publish_msg.type = "delete"
                                    publish_msg.current_state = self.suffix_state_sequence[suffix_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = 0.0
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    self.pose_index = self.previous_pose_index
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                            # self.get_logger().info("beanchmark fix 0.4")       
                            if check_in_bump(act, self.nodes, self.agent_name) and str(act) not in self.world.bump:
                                # self.get_logger().info("--------Bump detected---------")
                                self.world.bump[str(act)] = 1
                                # self.if_obs = True
                                try: 
                                    publish_msg = RelayRequest()
                                    publish_msg.type = "modify"
                                    publish_msg.current_state = self.suffix_state_sequence[suffix_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = self.actions[act]['weight']*5
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                        elif str(act) == "unload" or str(act) == "release":
                            self.mode = EquipmentMode.UNLOADED
                            self.act = 'u'
                        elif str(act) == "load":
                            for pt, label in self.task_points.items():
                                if abs(self.pose[0] - pt[0]) < 1e-6 and abs(self.pose[1] - pt[1]) < 1e-6:
                                    # Only publish UpdateValidTasks if this task is in the assigned task list
                                    if hasattr(self, 'cur_task_list') and label in self.cur_task_list:
                                        self.mode = EquipmentMode.LOADED
                                        msg = UpdateValidTasks()
                                        msg.robot_id = int(re.findall(r'\d+', self.agent_name)[0])
                                        msg.loaded_task = label
                                        self.update_valid_tasks_pub.publish(msg)
                                        self.get_logger().info(f'Published UpdateValidTasks: robot_id={self.agent_name}, loaded_task={msg.loaded_task}')
                                        self.act = 'l'
                                    break
                            self.previous_pose = self.pose
                            self.previous_pose_index = self.pose_index
                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
                        self.total_plan_index += 1
                        self.total_steps += 1  # Increment step counter
                        print(self.mode)
                        self.t = self.get_clock().now().to_msg()
                        self.next_interval = action_dict['weight']*5 # +1
                        
                        ########Logging
                        last_round_time = 50 if (self.previous_pose, self.pose) in self.world.bump else 10
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)-1):
                            pose_ab = extract_numbers(str(act))
                            # pose_a = extract_numbers(str(self.prefix_state_sequence[i].states[0]))
                            # pose_b = extract_numbers(str(self.prefix_state_sequence[i+1].states[0]))
                            if pose_ab not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                future_time += 10
                            else:
                                future_time += 50
                        if self.plan_index <= len(self.prefix_state_sequence):
                            for i in range(0, len(self.suffix_state_sequence)):
                                pose_ab = extract_numbers(str(act))
                                # pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                                # pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                                if pose_ab not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                    future_time += 10
                                else:
                                    future_time += 50
                        else:
                            for i in range(self.plan_index-len(self.prefix_state_sequence), len(self.suffix_state_sequence)):
                                pose_ab = extract_numbers(str(act))
                                # pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                                # pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                                if pose_ab not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                    future_time += 10
                                else:
                                    future_time += 50
                        self.pose_history.append((self.pose, round(self.next_interval+last_time), round(future_step), round(future_time)))
                        return
                    
            else:
                if self.pub_assign == True:
                    self.publish_task_request()
                    self.pub_assign = False
                    self.mode = EquipmentMode.WAITTASK
                    self.act = 's'


    def publish_task_request(self):
        robot_id = int(re.findall(r'\d+', self.agent_name)[0])
        task_request_msg = TaskRequestCluster()
        task_request_msg.robot_id = robot_id
        task_request_msg.task_status = 1
        task_request_msg.pose_index = self.pose_index
        task_request_msg.position = [float(x) for x in self.pose]

        
        self.taskassignment_request_pub.publish(task_request_msg)
        self.get_logger().info('------------Publish Task Assignment Request-------------')
    
    def transform_coords(self, coord):
        """Convert shapely coordinates to pygame coordinates."""
        x, y = coord
        return int(x * self.world.cell_size ), int(-y * self.world.cell_size + self.world.height)  # Flip y-axis for pygame

    
    def simulate(self):
        #rate = self.create_rate(10)
        
        try:    
            if (self.get_clock().now().nanoseconds - self.t_sim.nanoseconds) / 1e9 >= self.next_interval/20:      
                # self.get_logger().info(f"self.on_hold: {self.on_hold}")
                if self.on_hold == False and self.final_on_hold == False:
                    if USE_ISAAC:
                        if (self.sim_arrived or self.sim_received or self.sim_start):
                            # self.get_logger().info(f"Sim status: {self.sim_arrived}")
                            self.next_move()
                            self.sim_arrived = False
                            self.sim_received = False
                            self.sim_start = False
                            msg = AgentGoTo()
                            msg.agent_id = int(self.agent_name.replace("robot", ""))
                            msg.next_step = [float(x) for x in self.pose]
                            msg.next_flag = self.act
                            self.next_issac_step_pub.publish(msg)
                            self.get_logger().info(f'Next step published to Issac Sim...')
                    elif USE_HARDWARE:
                        # Hardware mode: TurtleBot3 with Nav2
                        # First, check if we have a pending goal that needs to be sent (Nav2 was not ready before)
                        if self.hardware_goal_pending and self.pending_goal_pose is not None:
                            self.get_logger().info(f'[{self.agent_name}] Retrying pending goal: ({self.pending_goal_pose[0]:.2f}, {self.pending_goal_pose[1]:.2f})')
                            self.send_nav2_goal(self.pending_goal_pose[0], self.pending_goal_pose[1])
                            # If goal was successfully sent, clear the pending flag
                            if not self.hardware_goal_pending:
                                self.get_logger().info(f'[{self.agent_name}] Pending goal successfully sent!')
                        # Only proceed if: goal reached, goal failed (need replan), or first start
                        elif (self.hardware_goal_reached or self.hardware_goal_failed or self.hardware_navigation_start):
                            if self.hardware_goal_failed:
                                # Navigation failed - may need to handle obstacle or retry
                                self.get_logger().warn(f'[{self.agent_name}] Hardware navigation failed! Attempting to continue...')
                            
                            # Execute the next move in the plan
                            self.next_move()
                            
                            # Reset flags
                            self.hardware_goal_reached = False
                            self.hardware_goal_failed = False
                            self.hardware_navigation_start = False
                            
                            # Send the new pose to TurtleBot3 via Nav2
                            # Only send navigation goal if we have a valid pose to go to
                            if self.pose != self.previous_pose or self.act == 'g':
                                self.send_nav2_goal(self.pose[0], self.pose[1])
                                self.get_logger().info(f'[{self.agent_name}] Nav2 goal sent to TurtleBot3: ({self.pose[0]:.2f}, {self.pose[1]:.2f})')
                            else:
                                # Non-movement action (load/unload), mark as immediately complete
                                self.hardware_goal_reached = True
                                self.get_logger().info(f'[{self.agent_name}] Non-movement action: {self.act}, proceeding...')
                    else:           
                        self.next_move()
                else:
                    replan_status_msg = ReplanStatus()
                    replan_status_msg.on_hold = self.on_hold
                    replan_status_msg.final_on_hold = self.final_on_hold

                    self.replan_pub.publish(replan_status_msg)
                self.t_sim = self.get_clock().now()    
                if self.pose != self.previous_pose:
                    if (self.previous_pose, self.pose) in self.world.bump or (self.pose, self.pose) in self.world.bump:
                        self.total_cost += 50
                    else:
                        self.total_cost += 10
                    try:
                        home_directory = os.path.expanduser("~")
                        with open(os.path.join(home_directory,'robot_data_10.csv'), mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(self.pose_history[-1])
                    except Exception as e:
                        print(f"An error occurred: {e}")
                # print("total_cost: ", self.total_cost)
               
            # =================================================================
            # pygame.draw.circle(self.world.screen, self.mode.value, (self.pose[0] * self.world.cell_size, \
            #     self.world.height - (self.pose[1] * self.world.cell_size)), self.world.cell_size // 5)
            # # self.world.screen.blit(text, text_)
            # =================================================================

            if self.mode == EquipmentMode.UNLOADED:
                mode = 'unloaded'
            elif self.mode == EquipmentMode.LOADED:
                mode = 'loaded'
            elif self.mode == EquipmentMode.WAITTASK:
                mode = 'Waiting'
                # self.get_logger().info(f"================Total Plan Index is {self.total_plan_index}.================")
            elif self.mode == EquipmentMode.NOTASK:
                mode = 'NoTask'
                # self.get_logger().info(f"================Total Plan Index is {self.total_plan_index}.================")
            elif self.mode == EquipmentMode.FAIL:
                mode = 'Fail'

            msg = ShowPosition()   # Create a new ShowPosition message instance
            msg.robot_id = self.agent_name
            msg.pose = [float(x) for x in self.pose]  # Convert tuple (1, 19) to list [1, 19] to match int32[] type
            msg.mode = mode
            self.position_pub.publish(msg)



            # pygame_surface = pygame.display.get_surface()
            # pygame_pixels = pygame.surfarray.array3d(pygame_surface)
            # image = np.flipud(pygame_pixels)

            # # Convert to BGR format (required by OpenCV)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # # Write the frame to the video file
            # self.world.output_video.write(image)
            
            # # Update the display
            # pygame.display.flip()

        except KeyboardInterrupt:
            print(self.pose_history)
            csv_file_name = "example.csv"
            with open(csv_file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                for row in self.pose_history:
                    writer.writerow(row)
        
#==============================
#             Main
#==============================
def main(args=None):
    # pygame.init()
    rclpy.init(args=args)
    node = rclpy.create_node('benchmark_node_main')

    grid_size = node.declare_parameter('N', 8).get_parameter_value().integer_value
    node.get_logger().info(f"grid_size: {grid_size}")
    
    env = GridWorld(grid_size)
    node.get_logger().info("reach here")
    ltl_drone = LTLControllerDrone(env)
    
    while(rclpy.ok()):
        try:
            # ltl_drone = LTLControllerDrone(env)
            rclpy.spin_once(ltl_drone)
        except ValueError as e:
            node.get_logger().error(f"LTL drone node: {e}")
            env.output_video.release()
            break

    # pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()