#!/usr/bin/env python
import os
import rclpy
from rclpy.node import Node
import sys
import yaml
import std_msgs
from copy import deepcopy
#Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TransitionSystemStateStamped, TransitionSystemState,UpdateValidTasks, WaitingRequest, StopWaiting, PositionRequest, TaskRequest, CurrentPosition, LTLPlan, RelayRequest, RelayResponse, ShowPosition
from ltl_automaton_msgs.srv import TaskReplanningDelete, TaskReplanningModify # TaskReplanningAddRequest, TaskReplanningDeleteRequest, TaskReplanningRelabelRequest
# Import transition system loader
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers, build_graph_halton, check_in_block, check_in_bump
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
#=================================================================
#  Interfaces between LTL planner node and lower level controls
#                       -----------------
# The node is reponsible for outputting current agent state based
# on TS and agent output.
# The node converts TS action to interpretable commands using
# action attributes defined in the TS config file
#=================================================================


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

class GridWorld(object):
    def __init__(self, grid_size):
         # Constants
        self.grid_size = grid_size
        self.load_elements()
        
        self.width, self.height = 800, 800
        self.cell_size = self.width // self.grid_size

        # Initialize the screen==================================
        # self.screen = pygame.display.set_mode((self.width, self.height))
        # self.font = pygame.font.SysFont('timesnewroman',  20)
        
        # self.frame_count = 0 
        # filename = "screen_%04d.png" % (self.frame_count)
        # pygame.image.save(self.screen, filename)
        # time.sleep(5)

        self.output_video = cv2.VideoWriter('/home/nanli/Isaac/planner/results/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))
        
        # pygame.display.set_caption("Grid with Moving Circle") # =====================

    
    def load_elements(self):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../src/lmco/ltl_automaton_planner'))
        # with open(parent_dir + '/config/benchmark_block_'+str(self.grid_size)+'.yaml', 'r') as file:
        with open(parent_dir + '/config/isaac_block.yaml', 'r') as file:
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
        with open(parent_dir + '/config/isaac_bump.yaml', 'r') as file:
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
    
    # def background(self):
    #     for key, pos in self.loc.items():
    #         letter = self.font.render(key, False, ORANGE, YELLOW)
    #         self.screen.blit(letter, (int(pos[0] * self.cell_size + self.cell_size/2), \
    #             self.height - int(pos[1] * self.cell_size + self.cell_size/2)))


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
            WaitingRequest,
            'no_task',
            self.no_task_callback,
            10
        )
        
        self.relay_pub = self.create_publisher(RelayRequest, 'replanning_request', 10)
        self.current_position_pub = self.create_publisher(CurrentPosition,'current_position', 10)
        self.update_pose_pub = self.create_publisher(CurrentPosition,'update_current_pose', 10)
        self.taskassignment_request_pub = self.create_publisher(TaskRequest, 'task_assignment_request', 10)
        self.position_pub = self.create_publisher(ShowPosition, 'show_position', 10)
        self.update_valid_tasks_pub = self.create_publisher(UpdateValidTasks, 'update_valid_tasks', 10)
        self.pub_assign = True
        self.on_hold = False
        self.final_on_hold = False
        self.no_task = False
        self.cur_task = 0
        
        transition_system_textfile = self.declare_parameter('transition_system_textfile', '').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        self.declare_parameter('agent_name', '')
        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        self.declare_parameter('init_state', 0)
        self.init_pose = self.get_parameter('init_state').value

        self.nodes, self.actions = build_graph_halton(20, 20, 700)
        self.transition_system ['state_models']['2d_pose_region']['nodes'] = self.nodes
        self.transition_system ['actions'].update(self.actions)

        self.mode = EquipmentMode.UNLOADED
        self.total_cost = 0
        self.if_obs = False

        if self.agent_name == 'robot_1':
            self.pose = (1, 19)
        elif self.agent_name =='robot_2':
            self.pose = (11, 19)
        elif self.agent_name =='robot_3':
            self.pose = (9, 11)
        elif self.agent_name =='robot_4':
            self.pose = (11, 9)
        

        self.pose_index = self.init_pose

        self.previous_pose = self.pose
        self.previous_pose_index = self.pose_index
        self.pose_history = [(self.pose, 0)]
        self.t_sim = self.get_clock().now()  # Use the ROS2 clock for the current time
        self.plan_index = 0
        self.next_interval = 10

        self.create_timer(1.0/10, self.simulate)
        # self.simulate()
    
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
        self.plan_index = 0
        self.cur_task = msg.cur_task
        self.world.block.clear()
        self.world.bump.clear()
        if self.final_on_hold == False:
            self.mode = EquipmentMode.UNLOADED
            self.on_hold = False
        else:
            self.mode = EquipmentMode.NOTASK
            # self.on_hold = True
        self.get_logger().info("receive data pre")
        self.prefix_action_list = msg.action_sequence
        self.get_logger().info(f"length prefix_action_list: {len(self.prefix_action_list)}")
        self.prefix_state_sequence = msg.ts_state_sequence
        self.get_logger().info("end data pre")
        self.get_logger().info(f'Prefix list received is {self.prefix_action_list}')
        
        # self.prefix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]

    def suffix_plan_callback(self, msg):
        # self.on_hold = False
        self.get_logger().info("receive data sub")
        self.suffix_action_list = msg.action_sequence
        self.get_logger().info(f"length suffix_action_list: {len(self.suffix_action_list)}")
        self.suffix_state_sequence = msg.ts_state_sequence
        self.get_logger().info("end data sub")
        self.get_logger().info(f'Suffix list received is {self.suffix_action_list}')
        
        # self.suffix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]
        
    def relay_callback(self, msg):
        self.get_logger().info("receive relay sub")
        self.pose = self.previous_pose
        if msg.success:
            self.prefix_action_list = msg.new_plan_prefix.action_sequence
            self.prefix_state_sequence = msg.new_plan_prefix.ts_state_sequence
            self.suffix_action_list = msg.new_plan_suffix.action_sequence
            self.suffix_state_sequence = msg.new_plan_suffix.ts_state_sequence
            self.on_hold = False

    def next_move(self):
        # if self.plan_index > 20 and self.pose == (grid_size/2-1, grid_size/2-1): #self.len(self.prefix_action_list) + len(self.suffix_action_list):
        #     sys.exit()
        #--------------
        # Move command
        #--------------
        # self.get_logger().info("inside the next move")
        if (len(self.prefix_action_list) + len(self.suffix_action_list) != 0):
            self.get_logger().info(f"plan index: {self.plan_index}")
            if self.plan_index < len(self.prefix_action_list):
                self.pub_assign = True
                #self.get_logger().info("beanchmark fix 0")
                for act in self.transition_system['actions']:
                    #self.get_logger().info("beanchmark fix 0.1")
                    if str(act) == self.prefix_action_list[self.plan_index]:
                        self.get_logger().info(str(act))
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
                            self.get_logger().info(f"previous pose: {self.previous_pose}")
                            self.get_logger().info(f"pose: {self.pose}")
                            
                            if check_in_block(act, self.nodes) and str(act) not in self.world.block:
                                self.get_logger().info("--------Block detected---------")
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
                            if check_in_bump(act, self.nodes) and str(act) not in self.world.bump:
                                self.get_logger().info("--------Bump detected---------")
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
                        elif str(act) == "load":
                            self.mode = EquipmentMode.LOADED
                            msg = UpdateValidTasks()
                            msg.robot_id = int(self.agent_name.split('_')[-1])
                            msg.loaded_task = self.cur_task
                            self.update_valid_tasks_pub.publish(msg)
                            self.get_logger().info(f'==================Published UpdateValidTasks: robot_id={self.agent_name}, loaded_task={msg.loaded_task}==================')
                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
                        self.get_logger().info(f"plan index: {self.plan_index}")
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
                            if check_in_block(act, self.nodes) and str(act) not in self.world.block:
                                self.get_logger().info("--------Block detected---------")
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
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    self.pose_index = self.previous_pose_index
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                            # self.get_logger().info("beanchmark fix 0.4")       
                            if check_in_bump(act, self.nodes) and str(act) not in self.world.bump:
                                self.get_logger().info("--------Bump detected---------")
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
                        elif str(act) == "unload" or str(act) == "release":
                            self.mode = EquipmentMode.UNLOADED
                        elif str(act) == "load":
                            self.mode = EquipmentMode.LOADED
                            msg = UpdateValidTasks()
                            msg.robot_id = int(self.agent_name.split('_')[-1])
                            msg.loaded_task = self.cur_task
                            self.update_valid_tasks_pub.publish(msg)
                            self.get_logger().info(f'Published UpdateValidTasks: robot_id={self.agent_name}, loaded_task={msg.loaded_task}')

                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
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


    def publish_task_request(self):
        robot_id = int(self.agent_name.split('_')[-1])
        task_request_msg = TaskRequest()
        task_request_msg.robot_id = robot_id
        task_request_msg.task_status = 1
        # task_request_msg.pose_index = self.pose_index

        
        self.taskassignment_request_pub.publish(task_request_msg)
        self.get_logger().info('------------Publish Task Assignment Request-------------')
    
    def transform_coords(self, coord):
        """Convert shapely coordinates to pygame coordinates."""
        x, y = coord
        return int(x * self.world.cell_size ), int(-y * self.world.cell_size + self.world.height)  # Flip y-axis for pygame

    
    def simulate(self):
        #rate = self.create_rate(10)
        
        # ====================================================================
        # lines = [LineString([(0, 3), (2, 3), (2, 4)]),
        #         LineString([(0, 5), (2, 5)]),
        #         LineString([(0, 7), (2, 7), (2, 6)]),
        #         LineString([(4, 9), (6, 9), (6, 10)]),
        #         LineString([(6, 7), (4, 7), (4, 5)]),
        #         LineString([(5, 5), (7, 5), (7, 7)]),
        #         LineString([(8, 5), (8, 7), (10, 7)]),
        #         LineString([(4, 2), (4, 4), (5, 4)]),
        #         LineString([(6, 4), (7, 4), (7, 2), (5, 2)]),

        #         LineString([(10, 3), (12, 3), (12, 4)]),
        #         LineString([(10, 5), (12, 5)]),
        #         LineString([(10, 7), (12, 7), (12, 6)]),
        #         LineString([(14, 9), (16, 9), (16, 10)]),
        #         LineString([(16, 7), (14, 7), (14, 5)]),
        #         LineString([(15, 5), (17, 5), (17, 7)]),
        #         LineString([(18, 5), (18, 7), (20, 7)]),
        #         LineString([(14, 2), (14, 4), (15, 4)]),
        #         LineString([(16, 4), (17, 4), (17, 2), (15, 2)]),

        #         LineString([(0, 13), (2, 13), (2, 14)]),
        #         LineString([(0, 15), (2, 15)]),
        #         LineString([(0, 17), (2, 17), (2, 16)]),
        #         LineString([(4, 19), (6, 19), (6, 20)]),
        #         LineString([(6, 17), (4, 17), (4, 15)]),
        #         LineString([(5, 15), (7, 15), (7, 17)]),
        #         LineString([(8, 15), (8, 17), (10, 17)]),
        #         LineString([(4, 12), (4, 14), (5, 14)]),
        #         LineString([(6, 14), (7, 14), (7, 12), (5, 12)]),
                
        #         LineString([(10, 13), (12, 13), (12, 14)]),
        #         LineString([(10, 15), (12, 15)]),
        #         LineString([(10, 17), (12, 17), (12, 16)]),
        #         LineString([(14, 19), (16, 19), (16, 20)]),
        #         LineString([(16, 17), (14, 17), (14, 15)]),
        #         LineString([(15, 15), (17, 15), (17, 17)]),
        #         LineString([(18, 15), (18, 17), (20, 17)]),
        #         LineString([(14, 12), (14, 14), (15, 14)]),
        #         LineString([(16, 14), (17, 14), (17, 12), (15, 12)]),
                
        #         LineString([(0, 10), (6, 10)]),
        #         LineString([(10, 0), (10, 7)]),
        #         LineString([(14, 10), (20, 10)]),
        #         LineString([(10, 13), (10, 20)])]

        # # Create buffered obstacles
        # obstacles = [line.buffer(distance=0.1, cap_style=3) for line in lines]
        
        
        
        # try:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             raise ValueError("pygame shutdown")

        #     # Clear the screen
        #     self.world.screen.fill(WHITE)
        #     # self.world.background()
        #     # Draw the grid
        #     # for row in range(3):
        #     #     for col in range(6):
        #     #         pygame.draw.rect(self.world.screen, BLACK, (col * self.world.cell_size, row * self.world.cell_size, self.world.cell_size, self.world.cell_size), 1)
            
        #     for obstacle in obstacles:
        #         if obstacle.geom_type == "Polygon":
        #             polygon_coords = [self.transform_coords(coord) for coord in obstacle.exterior.coords]
        #             pygame.draw.polygon(self.world.screen, RED, polygon_coords, 0)  # Filled polygon

            
        #     for action in self.actions:
        #         pose_ab = extract_numbers(str(action))
        #         pose_a = pose_ab[0]
        #         pose_b = pose_ab[1]
                
        #         start_pos = (
        #             int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
        #         )
        #         end_pos = (
        #             int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
        #         )

        #         pygame.draw.line(self.world.screen, BLACK, start_pos, end_pos, 1)  # 
            
        #     for action in self.world.block:
        #         pose_ab = extract_numbers(action)
        #         pose_a = pose_ab[0]
        #         pose_b = pose_ab[1]
                
        #         start_pos = (
        #             int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
        #         )
        #         end_pos = (
        #             int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
        #         )

        #         pygame.draw.line(self.world.screen, RED, start_pos, end_pos, 3)  # 
                
        #     for action in self.world.bump:
        #         pose_ab = extract_numbers(action)
        #         pose_a = pose_ab[0]
        #         pose_b = pose_ab[1]
                
        #         start_pos = (
        #             int(self.nodes[str(pose_a)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_a)]['attr']['pose'][1] * self.world.cell_size))
        #         )
        #         end_pos = (
        #             int(self.nodes[str(pose_b)]['attr']['pose'][0] * self.world.cell_size),
        #             int(self.world.height - (self.nodes[str(pose_b)]['attr']['pose'][1] * self.world.cell_size))
        #         )

        #         pygame.draw.line(self.world.screen, YELLOW, start_pos, end_pos, 3)  # 
            
        #     # Draw nodes
        #     for node in self.nodes:
        #         pygame.draw.circle(self.world.screen, BLUE, (self.nodes[node]['attr']['pose'][0]* self.world.cell_size, \
        #                        self.world.height - (self.nodes[node]['attr']['pose'][1]* self.world.cell_size)), 5)
            
            # ========================================================================
            
            # self.get_logger().info("inside b")    
            # # # Update to next action
            # self.get_logger.info(f"An error occurred: {e}")
        try:    
            if (self.get_clock().now().nanoseconds - self.t_sim.nanoseconds) / 1e9 >= self.next_interval/20:      
                # self.get_logger().info(f"self.on_hold: {self.on_hold}")
                if self.on_hold == False and self.final_on_hold == False:          
                    self.next_move()
                self.t_sim = self.get_clock().now()    
                if self.pose != self.previous_pose:
                    if (self.previous_pose, self.pose) in self.world.bump or (self.pose, self.pose) in self.world.bump:
                        self.total_cost += 50
                    else:
                        self.total_cost += 10
                    try:
                        with open('/home/nanli/robot_data_10.csv', mode='a', newline='') as file:
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
            elif self.mode == EquipmentMode.NOTASK:
                mode = 'NoTask'

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