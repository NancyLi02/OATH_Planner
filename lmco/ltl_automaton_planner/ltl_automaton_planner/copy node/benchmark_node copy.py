#!/usr/bin/env python
import os
import rclpy
from rclpy.node import Node
import sys
import yaml
import std_msgs
from copy import deepcopy
#Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TransitionSystemStateStamped, TransitionSystemState, PositionRequest, TaskRequest, CurrentPosition, LTLPlan, RelayRequest, RelayResponse
from ltl_automaton_msgs.srv import TaskReplanningDelete, TaskReplanningModify # TaskReplanningAddRequest, TaskReplanningDeleteRequest, TaskReplanningRelabelRequest
# Import transition system loader
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers
# Import modules for commanding the a1
import re
from geometry_msgs.msg import Point
from std_msgs.msg import String, Bool
import pygame
from enum import Enum
import cv2
import numpy as np
import time
import csv
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
    RESCUE = (255, 0, 0)
    WAITTASK = (255, 100, 0)

class GridWorld(object):
    def __init__(self, grid_size):
         # Constants
        self.grid_size = grid_size
        self.load_elements()
        
        self.width, self.height = 780, 780
        self.cell_size = self.width // self.grid_size

        # Initialize the screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont('timesnewroman',  20)
        
        # self.frame_count = 0 
        # filename = "screen_%04d.png" % (self.frame_count)
        # pygame.image.save(self.screen, filename)
        # time.sleep(5)

        self.output_video = cv2.VideoWriter('/home/nanli/Isaac/planner/results/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))
        
        pygame.display.set_caption("Grid with Moving Circle")

    
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
        
        self.relay_pub = self.create_publisher(RelayRequest, 'replanning_request', 10)
        self.current_position_pub = self.create_publisher(CurrentPosition,'current_position', 10)
        self.taskassignment_request_pub = self.create_publisher(TaskRequest, 'task_assignment_request', 10)
        self.pub_assign = True
        self.on_hold = False

        # self.delete_client = self.create_client(TaskReplanningDelete, 'replanning_delete')
        # if not self.delete_client.wait_for_service(timeout_sec=1000.0):  # Set your desired timeout in seconds
        #     self.get_logger().error('Service /replanning_delete not available after waiting')
        # else:
        #     self.get_logger().info('Service /replanning_delete is available')
            
        # self.modify_client = self.create_client(TaskReplanningModify, 'replanning_modify')
        # if not self.modify_client.wait_for_service(timeout_sec=1000.0):  # Set your desired timeout in seconds
        #     self.get_logger().error('Service /replanning_modify not available after waiting')
        # else:
        #     self.get_logger().info('Service /replanning_modify is available')
        
        transition_system_textfile = self.declare_parameter('transition_system_textfile', '').get_parameter_value().string_value
        self.transition_system = import_ts_from_file(transition_system_textfile)
        #print(self.transition_system)
        self.declare_parameter('agent_name', '')
        self.agent_name = self.get_parameter('agent_name').get_parameter_value().string_value
        
        self.mode = EquipmentMode.UNLOADED
        self.total_cost = 0
        self.if_obs = False
        if self.agent_name == 'robot_1':
            self.pose = (0, 0)
        elif self.agent_name =='robot_2':
            self.pose = (4, 3)
        self.previous_pose = self.pose
        self.pose_history = [(self.pose, 0)]
        self.t_sim = self.get_clock().now()  # Use the ROS2 clock for the current time
        self.plan_index = 0
        self.next_interval = 10
        # self.get_logger().info("BN sending request")    
        # response = self.send_request(2, 3)
        # self.get_logger().info(
        #         'BN Result of add_two_ints: for %d + %d = %d' %
        #         (20, 32, response.sum))
        self.create_timer(1.0/10, self.simulate)
        # self.simulate()
    
    def get_current_pos(self, msg=None):
        self.get_logger().info('Getting Current Position Now ................................')
        current_pos_msg = CurrentPosition()
        current_pos_msg.robot_id = self.agent_name
        current_pos_msg.position = list(self.pose)
        if self.mode == EquipmentMode.LOADED:
            current_pos_msg.current_state = 'loaded'
        else:
            current_pos_msg.current_state = 'unloaded'
        self.current_position_pub.publish(current_pos_msg)
        self.get_logger().info(f"Publishing current position for {self.agent_name}: {self.pose}, with state {current_pos_msg.current_state}.")

    def prefix_plan_callback(self, msg):
        self.plan_index = 0
        self.mode = EquipmentMode.UNLOADED
        self.get_logger().info("receive data pre")
        self.prefix_action_list = msg.action_sequence
        self.get_logger().info(f"length prefix_action_list: {len(self.prefix_action_list)}")
        self.prefix_state_sequence = msg.ts_state_sequence
        self.get_logger().info("end data pre")
        # self.prefix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]

    def suffix_plan_callback(self, msg):
        self.get_logger().info("receive data sub")
        self.suffix_action_list = msg.action_sequence
        self.get_logger().info(f"length suffix_action_list: {len(self.suffix_action_list)}")
        self.suffix_state_sequence = msg.ts_state_sequence
        self.get_logger().info("end data sub")
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
            # self.get_logger().info(f"plan index: {self.plan_index}")
            if self.plan_index < len(self.prefix_action_list):
                self.pub_assign = True
                #self.get_logger().info("beanchmark fix 0")
                for act in self.transition_system['actions']:
                    #self.get_logger().info("beanchmark fix 0.1")
                    if str(act) == self.prefix_action_list[self.plan_index]:
                        #self.get_logger().info("beanchmark fix 0.2")
                        # Extract action types, attributes, etc. in dictionary
                        action_dict = self.transition_system['actions'][str(act)]
                        # print(self.prefix_action_list)
                        if str(act)[:4] == "goto":
                            # self.get_logger().info("beanchmark fix 0.3")
                            self.previous_pose = self.pose
                            self.pose = extract_numbers(str(act))
                            # self.get_logger().info(f"previous pose: {self.previous_pose}")
                            # self.get_logger().info(f"pose: {self.pose}")
                            if (self.previous_pose, self.pose) in self.world.wall or (self.pose, self.previous_pose) in self.world.wall:
                                self.get_logger().info("--------Obstacle detected---------")
                                self.world.wall[((self.previous_pose, self.pose))] = 1
                                self.world.wall[((self.pose, self.previous_pose))] = 1
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
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                            # self.get_logger().info("beanchmark fix 0.4")       
                            if self.world.bump.get((self.previous_pose, self.pose)) == 0 or self.world.bump.get((self.pose, self.previous_pose)) == 0 :
                                self.get_logger().info("--------Obstacle detected---------")
                                self.world.bump[((self.previous_pose, self.pose))] = 1
                                self.world.bump[((self.pose, self.previous_pose))] = 1
                                # self.if_obs = True
                                try: 
                                    publish_msg = RelayRequest()
                                    publish_msg.type = "modify"
                                    publish_msg.current_state = self.prefix_state_sequence[self.plan_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = 50.0
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
                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
                        self.get_logger().info(f"plan index: {self.plan_index}")
                        print(self.mode)
                        self.t = self.get_clock().now().to_msg()
                        self.next_interval = action_dict['weight'] # +1
                        # self.get_logger().info("beanchmark fix 0.6")
                        ##### Logging
                        last_round_time = 50 if (self.previous_pose, self.pose) in self.world.bump else 10
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)-1):
                            pose_a = extract_numbers(str(self.prefix_state_sequence[i].states[0]))
                            pose_b = extract_numbers(str(self.prefix_state_sequence[i+1].states[0]))
                            if (pose_a, pose_b) not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                future_time += 10
                            else:
                                future_time += 50
                        for i in range(0, len(self.suffix_state_sequence)):
                            pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                            pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                            if (pose_a, pose_b) not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
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
                            self.pose = extract_numbers(str(act))
                            if (self.previous_pose, self.pose) in self.world.wall or (self.pose, self.previous_pose) in self.world.wall:
                                print("\n")
                                print("----Obstacle detected----")
                                self.world.wall[((self.previous_pose, self.pose))] = 1
                                self.world.wall[((self.pose, self.previous_pose))] = 1
                                self.if_obs = True
                                try: 
                                    publish_msg = RelayRequest()
                                    publish_msg.type = "delete"
                                    publish_msg.current_state = self.prefix_state_sequence[self.plan_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = 0.0
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                                    
                            if self.world.bump.get((self.previous_pose, self.pose)) == 0 or self.world.bump.get((self.pose, self.previous_pose)) == 0 :
                                print("\n")
                                print("----Bump detected----")
                                self.world.bump[((self.previous_pose, self.pose))] = 1
                                self.world.bump[((self.pose, self.previous_pose))] = 1
                                # self.if_obs = True
                                try: 
                                    publish_msg = RelayRequest()
                                    publish_msg.type = "modify"
                                    publish_msg.current_state = self.prefix_state_sequence[self.plan_index]
                                    publish_msg.from_pose.extend(list(self.previous_pose))
                                    publish_msg.to_pose.extend(list(self.pose))
                                    publish_msg.exec_index = self.plan_index
                                    publish_msg.cost = 50.0
                                    self.relay_pub.publish(publish_msg)
                                    self.on_hold = True
                                    self.pose = self.previous_pose
                                    return
                                except Exception as e:
                                    self.get_logger().error(f'Failed to call service: {e}')
                                    exit(1)
                                    
                            # self.pose = (int(str(act).split('c')[1]), int(str(act).split('r')[1]))
                        elif str(act) == "unload" or str(act) == "release":
                            self.mode = EquipmentMode.UNLOADED
                        elif str(act) == "load":
                            self.mode = EquipmentMode.LOADED
                        elif str(act) == "goto_rescue":
                            self.mode = EquipmentMode.RESCUE
                        else: # including action "stay", nothing particular needs to be done
                            pass
                        self.plan_index += 1
                        print(self.mode)
                        self.t = self.get_clock().now().to_msg()
                        self.next_interval = action_dict['weight'] # +1
                        
                        ########Logging
                        last_round_time = 50 if (self.previous_pose, self.pose) in self.world.bump else 10
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)-1):
                            pose_a = extract_numbers(str(self.prefix_state_sequence[i].states[0]))
                            pose_b = extract_numbers(str(self.prefix_state_sequence[i+1].states[0]))
                            if (pose_a, pose_b) not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                future_time += 10
                            else:
                                future_time += 50
                        if self.plan_index <= len(self.prefix_state_sequence):
                            for i in range(0, len(self.suffix_state_sequence)):
                                pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                                pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                                if (pose_a, pose_b) not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
                                    future_time += 10
                                else:
                                    future_time += 50
                        else:
                            for i in range(self.plan_index-len(self.prefix_state_sequence), len(self.suffix_state_sequence)):
                                pose_a = extract_numbers(str(self.suffix_state_sequence[i].states[0]))
                                pose_b = extract_numbers(str(self.suffix_state_sequence[(i+1)%len(self.suffix_state_sequence)].states[0]))
                                if (pose_a, pose_b) not in self.world.bump or self.world.bump.get((self.previous_pose, self.pose)) == 0:
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
        task_request_msg.position_x = self.pose[0]
        task_request_msg.position_y = self.pose[1]
        
        self.taskassignment_request_pub.publish(task_request_msg)
        self.get_logger().info('------------Publish Task Assignment Request-------------')


    def simulate(self):
        #rate = self.create_rate(10)
        
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise ValueError("pygame shutdown")

            # Clear the screen
            self.world.screen.fill(WHITE)
            # self.world.background()
            # Draw the grid
            for row in range(self.world.grid_size):
                for col in range(self.world.grid_size):
                    pygame.draw.rect(self.world.screen, BLACK, (col * self.world.cell_size, row * self.world.cell_size, self.world.cell_size, self.world.cell_size), 1)
            # self.get_logger().info("inside b")    
            # # # Update to next action
            # self.get_logger.info(f"An error occurred: {e}")
            if (self.get_clock().now().nanoseconds - self.t_sim.nanoseconds) / 1e9 >= self.next_interval/50:      
                # self.get_logger().info(f"self.on_hold: {self.on_hold}")
                if self.on_hold == False:             
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
            
            # Draw the moving agent
            # text = pygame.font.SysFont('timesnewroman', 10).render(str(self.mode), True, BLACK, WHITE)
            # text_ = text.get_rect()
            # text_.center = (int(self.pose[0] * self.world.cell_size + self.world.cell_size/2), \
            #     self.world.height - int(self.pose[1] * self.world.cell_size + self.world.cell_size/2))
            # self.get_logger().info(f"Pose in simulate: x={self.pose[0]}, y={self.pose[1]}")
            # self.get_logger().info("----------------------------------------------------------------------------------")
            # for i in range(1, self.world.grid_size-1):
            #     pygame.draw.line(self.world.screen, BLACK,  
            #                 (self.world.grid_size/2*self.world.cell_size, (i+1)*self.world.cell_size),\
            #                 ((self.world.grid_size/2)*self.world.cell_size, i*self.world.cell_size), 5)
            #     pygame.draw.line(self.world.screen, BLACK,  
            #                 (i*self.world.cell_size, self.world.grid_size/2*self.world.cell_size),\
            #                 ((i+1)*self.world.cell_size, (self.world.grid_size/2)*self.world.cell_size), 5)
            
            for wall, checked in self.world.wall.items():
                if checked:
                    if wall[0][1] == wall[1][1]: # vertical
                        pygame.draw.line(self.world.screen, RED,  
                            (max(wall[0][0], wall[1][0])*self.world.cell_size, self.world.height - (wall[0][1]+1)*self.world.cell_size),\
                            (max(wall[0][0], wall[1][0])*self.world.cell_size, self.world.height - wall[0][1]*self.world.cell_size), 5)
                    if wall[0][0] == wall[1][0]: # horizontal
                        pygame.draw.line(self.world.screen, RED,  
                            (wall[0][0]*self.world.cell_size, self.world.height - max(wall[0][1], wall[1][1])*self.world.cell_size),\
                            ((wall[0][0]+1)*self.world.cell_size, self.world.height - max(wall[0][1], wall[1][1])*self.world.cell_size), 5)
            
            for wall, checked in self.world.bump.items():
                if checked:
                    if wall[0][1] == wall[1][1]: # vertical
                        pygame.draw.line(self.world.screen, YELLOW,  
                            (max(wall[0][0], wall[1][0])*self.world.cell_size, self.world.height - (wall[0][1]+1)*self.world.cell_size),\
                            (max(wall[0][0], wall[1][0])*self.world.cell_size, self.world.height - wall[0][1]*self.world.cell_size), 5)
                    if wall[0][0] == wall[1][0]: # horizontal
                        pygame.draw.line(self.world.screen, YELLOW,  
                            (wall[0][0]*self.world.cell_size, self.world.height - max(wall[0][1], wall[1][1])*self.world.cell_size),\
                            ((wall[0][0]+1)*self.world.cell_size, self.world.height - max(wall[0][1], wall[1][1])*self.world.cell_size), 5)
                    
            
            pygame.draw.circle(self.world.screen, self.mode.value, (int(self.pose[0] * self.world.cell_size + self.world.cell_size/2), \
                self.world.height - int(self.pose[1] * self.world.cell_size + self.world.cell_size/2)), self.world.cell_size // 2)
            # self.world.screen.blit(text, text_)
            
            
            pygame_surface = pygame.display.get_surface()
            pygame_pixels = pygame.surfarray.array3d(pygame_surface)
            image = np.flipud(pygame_pixels)

            # Convert to BGR format (required by OpenCV)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Write the frame to the video file
            self.world.output_video.write(image)
            
            # Update the display
            pygame.display.flip()
            # self.get_logger().info("inside d")    

            # if not (self.curr_ltl_state == self.prev_ltl_state):
            #     # Update previous state
            #     self.prev_ltl_state = deepcopy(self.curr_ltl_state)
            #     # If all states are initialized (not None), publish message
            #     if all([False for element in self.curr_ltl_state if element == None]):
            #         # Publish msg
            #         self.ltl_state_msg.header.stamp = rospy.Time.now()
            #         self.ltl_state_msg.ts_state.states = self.curr_ltl_state
            #         self.ltl_state_pub.publish(self.ltl_state_msg)

            # If waiting for obstacles or acknowledgement, check again
            # if self.next_action:
            #     # If action returns true, action was carried out and is reset
            #     if self.a1_action(self.next_action):
            #         self.a1_action = {}
                    
            # rospy.loginfo("State is %s and prev state is %s" %(self.curr_ltl_state, self.prev_ltl_state))
            # rate.sleep()    
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
    pygame.init()
    rclpy.init(args=args)
    node = rclpy.create_node('benchmark_node_main')

    grid_size = node.declare_parameter('N', 10).get_parameter_value().integer_value
    node.get_logger().info(f"grid_size: {grid_size}")
    
    env = GridWorld(grid_size)
    node.get_logger().info("reach here")
    ltl_drone = LTLControllerDrone(env)
    
    while(rclpy.ok()):
        try:
            rclpy.spin_once(ltl_drone)
        except ValueError as e:
            node.get_logger().error(f"LTL drone node: {e}")
            env.output_video.release()
            break

    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()