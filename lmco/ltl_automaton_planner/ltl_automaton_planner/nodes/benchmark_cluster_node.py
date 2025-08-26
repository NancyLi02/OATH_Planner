#!/usr/bin/env python
import os
import rclpy
from rclpy.node import Node
import sys
import yaml
import std_msgs
from copy import deepcopy
#Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TaskFail, AgentFail, AgentFailTask, NoTask, TransitionSystemStateStamped, TransitionSystemState,UpdateValidTasks, WaitingRequest, StopWaiting, PositionRequest, TaskRequestCluster, CurrentPosition, LTLPlan, RelayRequest, RelayResponse, ShowPosition, AddTask, ObstacleUpdate
from ltl_automaton_msgs.srv import TaskReplanningDelete, TaskReplanningModify # TaskReplanningAddRequest, TaskReplanningDeleteRequest, TaskReplanningRelabelRequest
# Import transition system loader
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers, build_graph_halton, check_in_block, check_in_bump, add_block_polygon, add_bump_polygon, update_graph_with_obstacle
# Import modules for commanding the a1

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
import pygame
from enum import Enum
import cv2
import numpy as np
import time
import csv
from shapely.geometry import Point, LineString, Polygon
from example_interfaces.srv import AddTwoInts
import re
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from interfaces_hmm_sim.msg import Status, ReplanStatus, AgentGoTo
from ament_index_python.packages import get_package_share_directory

#=================================================================
#  Interfaces between LTL planner node and lower level controls
#                       -----------------
# The node is reponsible for outputting current agent state based
# on TS and agent output.
# The node converts TS action to interpretable commands using
# action attributes defined in the TS config file
#=================================================================

USE_ISAAC = False

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

        self.nodes, self.actions = build_graph_halton(20, 20, 1000)
        
        self.transition_system ['state_models']['2d_pose_region']['nodes'] = self.nodes
        self.transition_system ['actions'].update(self.actions)

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

        # 从yaml文件加载任务点和pickup/delivery映射
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
        obstacle_key = tuple(msg.obstacle_location)
        if obstacle_key in self.processed_obstacles:
            self.get_logger().info(f"Ignoring duplicate obstacle update for location: {msg.obstacle_location}")
            return
            
        self.get_logger().info(f"Received obstacle update: {msg.obstacle_type} at {msg.obstacle_location}")
        self.processed_obstacles.add(obstacle_key)

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
            # add_block_polygon(vertices) # For visualization consistency
            update_graph_with_obstacle(self.nodes, self.actions, new_obstacle)
            self.get_logger().info("Graph updated locally. Checking current path for collision.")
            self.check_path_for_new_obstacle()
        elif msg.obstacle_type == 'bush' or msg.obstacle_type == 'bump':
            add_bump_polygon(vertices)
            self.get_logger().info("New bump obstacle added.")
            # Bumps add cost, but don't block, so we might not need to replan immediately unless the path becomes too costly
            # Or we can just let the existing bump logic handle it on the next move.
        
    def check_path_for_new_obstacle(self):
        # We need to check if the current or upcoming path segment is now blocked.
        # This checks the entire remaining prefix plan.
        if (len(self.prefix_action_list) == 0):
            return

        plan_is_valid = True
        for i in range(self.plan_index, len(self.prefix_action_list)):
            action_to_check = self.prefix_action_list[i]

            if not str(action_to_check).startswith("from_"):
                continue

            # Check if the action itself has been removed from the graph
            if action_to_check not in self.actions:
                self.get_logger().warn(f"Path invalidated: Action '{action_to_check}' no longer exists in the graph.")
                plan_is_valid = False
                break
            
            # Legacy check for nodes (belt-and-suspenders)
            from_idx, to_idx = extract_numbers(str(action_to_check))
            if str(from_idx) not in self.nodes or str(to_idx) not in self.nodes:
                self.get_logger().warn(f"Path invalidated: Node for action '{action_to_check}' was removed.")
                plan_is_valid = False
                break
        
        if not plan_is_valid:
            self.get_logger().info("Current plan is no longer valid due to new obstacle. Requesting new task assignment.")
            # Stop current movement and request a new plan/task from the central planner
            self.on_hold = True
            # Clearing the plan will prevent further movement and the logic will naturally
            # lead to a new task request after the current (now empty) plan is "finished".
            self.prefix_action_list = []
            self.suffix_action_list = []
            self.publish_task_request()


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
            self.nodes, self.actions = build_graph_halton(20, 20, 1000, self.new_task_points)
            self.transition_system['state_models']['2d_pose_region']['nodes'] = self.nodes
            self.transition_system['actions'].update(self.actions)
            self.get_logger().info("Transition system and task points updated.")
            # Clear the list after updating
            self.new_task_points.clear()

        self.plan_index = 0
        self.cur_task_list = msg.route_labels
        self.i = 0
        self.cur_task = self.cur_task_list[self.i]
        # self.world.block.clear()
        # self.world.bump.clear()
        if self.final_on_hold == False:
            self.mode = EquipmentMode.UNLOADED
            self.on_hold = False
        else:
            self.mode = EquipmentMode.NOTASK
            # self.on_hold = True
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
        # self.get_logger().info("receive relay sub")
        self.pose = self.previous_pose
        if msg.success:
            self.prefix_action_list = msg.new_plan_prefix.action_sequence
            self.prefix_state_sequence = msg.new_plan_prefix.ts_state_sequence
            self.suffix_action_list = msg.new_plan_suffix.action_sequence
            self.suffix_state_sequence = msg.new_plan_suffix.ts_state_sequence
            self.on_hold = False

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
                            # 只有到达任务点时才发布UpdateValidTasks
                            for pt, label in self.task_points.items():
                                if abs(self.pose[0] - pt[0]) < 1e-6 and abs(self.pose[1] - pt[1]) < 1e-6:
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
                self.get_logger().info(f"================Total Plan Index is {self.total_plan_index}.================")
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