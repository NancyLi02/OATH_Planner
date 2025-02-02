#!/usr/bin/env python
import rospy
import sys
import yaml
import std_msgs
from copy import deepcopy
#Import LTL automaton message definitions
from ltl_automaton_msgs.msg import TransitionSystemStateStamped, TransitionSystemState, LTLPlan
from ltl_automaton_msgs.srv import * # TaskReplanningAddRequest, TaskReplanningDeleteRequest, TaskReplanningRelabelRequest
# Import transition system loader
from ltl_automaton_planner.ltl_automaton_utilities import import_ts_from_file, extract_numbers
# Import modules for commanding the a1
import actionlib
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
import pygame
from enum import Enum
import cv2
import numpy as np
import time
import csv
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


class GridWorld(object):
    def __init__(self):
         # Constants
        self.width, self.height = 780, 780
        self.grid_size = 20
        self.cell_size = self.width // self.grid_size

        # Initialize the screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font=pygame.font.SysFont('timesnewroman',  20)
        
        # self.frame_count = 0 
        # filename = "screen_%04d.png" % (self.frame_count)
        # pygame.image.save(self.screen, filename)
        time.sleep(5)
        self.output_video = cv2.VideoWriter('/home/haris/Isaac/planner/results/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))

        self.loc = {'C': (0, 1),
                    'P3': (0, 3),
                    #'I': (1, 4),
                    'H': (2, 2),
                    'P4': (5, 2),
                    #'P1': (3, 0),
                    'L1': (3, 5),
                    #'F': (4, 3),
                    #'L2': (5, 0),
                    #'P2': (5, 2)
                    }
        self.wall = {((0, 5), (1, 5)): 0,
                     ((2, 3), (2, 2)): 0,
                     ((3, 1), (4, 1)): 0,
                     ((3, 2), (4, 2)): 0,
                     ((3, 3), (4, 3)): 0,
                     ((2, 5), (3, 5)): 0,
                     ((3, 4), (3, 5)): 0,
                     ((4, 5), (3, 5)): 0
                     }
        pygame.display.set_caption("Grid with Moving Circle")
        # pygame.time.delay(10000) 
        
    def background(self):
        for key, pos in self.loc.items():
            letter = self.font.render(key, False, ORANGE, YELLOW)
            self.screen.blit(letter, (int(pos[0] * self.cell_size + self.cell_size/2), \
                self.height - int(pos[1] * self.cell_size + self.cell_size/2)))
            

# class GridWorld(object):
#     def __init__(self):
#          # Constants
#         self.width, self.height = 160, 160
#         self.grid_size = 2
#         self.cell_size = self.width // self.grid_size

#         # Initialize the screen
#         self.screen = pygame.display.set_mode((self.width, self.height))
#         self.font=pygame.font.SysFont('timesnewroman',  20)
        
#         # self.frame_count = 0 
#         # filename = "screen_%04d.png" % (self.frame_count)
#         # pygame.image.save(self.screen, filename)
#         self.output_video = cv2.VideoWriter('/home/jren313/Downloads/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))

#         self.loc = {'A': (0, 1),
#                     'B': (1, 1),
#                     }
#         self.wall = {((0, 1), (1, 1)): 0,
#                      ((1, 0), (1, 1)): 0}
#         pygame.display.set_caption("Grid with Moving Circle")
        
#     def background(self):
#         for key, pos in self.loc.items():
#             letter = self.font.render(key, False, ORANGE, YELLOW)
#             self.screen.blit(letter, (int(pos[0] * self.cell_size + self.cell_size/2), \
#                 self.height - int(pos[1] * self.cell_size + self.cell_size/2)))


class LTLControllerDrone(object):
    def __init__(self, env):
        self.world = env
        self.action_list = []
        
        self.prefix_action_list = []
        self.suffix_action_list = []
        self.drone_prefix_sub = rospy.Subscriber('/prefix_plan', LTLPlan, self.prefix_plan_callback)
        self.drone_suffix_sub = rospy.Subscriber('/suffix_plan', LTLPlan, self.suffix_plan_callback, queue_size=100)
        
        transition_system_textfile = rospy.get_param('transition_system_textfile')
        self.transition_system = import_ts_from_file(transition_system_textfile)
        #print(self.transition_system)
        
        self.mode = EquipmentMode.UNLOADED
        self.if_obs = False
        self.pose = (0, 1)
        self.pose_history = [(self.pose, 0)]
        self.simulate()
    
    def prefix_plan_callback(self, msg):
        self.prefix_action_list = msg.action_sequence
        self.prefix_state_sequence = msg.ts_state_sequence
        # self.prefix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]

    def suffix_plan_callback(self, msg):
        self.suffix_action_list = msg.action_sequence
        self.suffix_state_sequence = msg.ts_state_sequence
        # self.suffix_action_list = [(int(s.split('c')[1]), int(s.split('r')[1])) for s in action_seq]

    
    def next_move(self):
        #--------------
        # Move command
        #--------------
        if (len(self.prefix_action_list) + len(self.suffix_action_list) != 0):
            if self.plan_index < len(self.prefix_action_list):
                for act in self.transition_system['actions']:
                    if str(act) == self.prefix_action_list[self.plan_index]:
                        # Extract action types, attributes, etc. in dictionary
                        action_dict = self.transition_system['actions'][str(act)]
                        # print(self.prefix_action_list)
                        if str(act)[:4] == "goto":
                            self.previous_pose = self.pose
                            self.pose = extract_numbers(str(act))
                            if (self.previous_pose, self.pose) in self.world.wall or (self.pose, self.previous_pose) in self.world.wall:
                                print("----Obstacle detected----")
                                self.world.wall[((self.previous_pose, self.pose))] = 1
                                self.world.wall[((self.pose, self.previous_pose))] = 1
                                # self.if_obs = True
                                try: 
                                    service = rospy.ServiceProxy('/replanning_delete', TaskReplanningDelete)
                                    task_replanning_srv = TaskReplanningDeleteRequest()
                                    # print(self.prefix_state_sequence[self.plan_index].states)
                                    task_replanning_srv.current_state = self.prefix_state_sequence[self.plan_index]
                                    # task_replanning_srv.delete_from = list()
                                    # task_replanning_srv.delete_to = list()
                                    task_replanning_srv.delete_from.extend(list(self.previous_pose))
                                    task_replanning_srv.delete_to.extend(list(self.pose))
                                    task_replanning_srv.exec_index = self.plan_index
                                    response = service(task_replanning_srv)
                                    if response.success:
                                        self.pose = self.previous_pose
                                        self.prefix_plan_callback(response.new_plan_prefix)
                                        self.suffix_plan_callback(response.new_plan_suffix)
                                        return
                                except rospy.ServiceException as e:
                                    print(f"Service call failed: {e}")
                                    exit(1)
                                    
                            if (self.previous_pose, self.pose) in self.world.bump or (self.pose, self.previous_pose) in self.world.bump:
                                print("----Bump detected----")
                                self.world.bump[((self.previous_pose, self.pose))] = 1
                                self.world.bump[((self.pose, self.previous_pose))] = 1
                                # self.if_obs = True
                                try: 
                                    service = rospy.ServiceProxy('/replanning_mod', TaskReplanningModify)
                                    task_replanning_srv = TaskReplanningModifyRequest()
                                    # print(self.prefix_state_sequence[self.plan_index].states)
                                    task_replanning_srv.current_state = self.prefix_state_sequence[self.plan_index]
                                    # task_replanning_srv.delete_from = list()
                                    # task_replanning_srv.delete_to = list()
                                    task_replanning_srv.mod_from.extend(list(self.previous_pose))
                                    task_replanning_srv.mod_to.extend(list(self.pose))
                                    task_replanning_srv.exec_index = self.plan_index
                                    task_replanning_srv.cost = 30
                                    response = service(task_replanning_srv)
                                    if response.success:
                                        self.pose = self.previous_pose
                                        self.prefix_plan_callback(response.new_plan_prefix)
                                        self.suffix_plan_callback(response.new_plan_suffix)
                                        return
                                except rospy.ServiceException as e:
                                    print(f"Service call failed: {e}")
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
                        self.t = rospy.Time.now()
                        self.next_interval = action_dict['weight'] # +1
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)):
                            for act in self.transition_system['actions']:
                                if str(act) == self.prefix_action_list[self.plan_index]:
                                    future_time += self.transition_system['actions']["weight"]
                        for i in range(0, len(self.suffix_state_sequence)):
                            for act in self.transition_system['actions']:
                                if str(act) == self.prefix_action_list[self.plan_index]:
                                    future_time += self.transition_system['actions']["weight"]
                        self.pose_history.append((self.pose, self.next_interval+last_time, future_step, future_time))
                        return

            if self.plan_index >= len(self.prefix_action_list):
                suffix_index = (self.plan_index - len(self.prefix_action_list)) % len(self.suffix_action_list)
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
                                print("----Obstacle detected----")
                                self.world.wall[((self.previous_pose, self.pose))] = 1
                                self.world.wall[((self.pose, self.previous_pose))] = 1
                                self.if_obs = True
                                try: 
                                    service = rospy.ServiceProxy('/replanning_delete', TaskReplanningDelete)
                                    task_replanning_srv = TaskReplanningDeleteRequest()
                                    # print(self.suffix_state_sequence[self.plan_index].states)
                                    task_replanning_srv.current_state = self.suffix_state_sequence[suffix_index]
                                    # task_replanning_srv.delete_from = list()
                                    # task_replanning_srv.delete_to = list()
                                    task_replanning_srv.delete_from.extend(list(self.previous_pose))
                                    task_replanning_srv.delete_to.extend(list(self.pose))
                                    task_replanning_srv.exec_index = self.plan_index
                                    response = service(task_replanning_srv)
                                    if response.success:
                                        self.pose = self.previous_pose
                                        self.prefix_plan_callback(response.new_plan_prefix)
                                        self.suffix_plan_callback(response.new_plan_suffix)
                                        print(self.pose)
                                        # self.plan_index -= 1
                                        return
                                except rospy.ServiceException as e:
                                    print(f"Service call failed: {e}")
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
                        self.t = rospy.Time.now()
                        self.next_interval = action_dict['weight'] # +1
                        last_time = self.pose_history[-1][1]
                        future_step = len(self.prefix_state_sequence) + len(self.suffix_state_sequence) - self.plan_index
                        future_time = 0
                        for i in range(self.plan_index, len(self.prefix_state_sequence)):
                            for act in self.transition_system['actions']:
                                if str(act) == self.prefix_action_list[self.plan_index]:
                                    future_time += self.transition_system['actions']["weight"]
                        for i in range(0, len(self.suffix_state_sequence)):
                            for act in self.transition_system['actions']:
                                if str(act) == self.prefix_action_list[self.plan_index]:
                                    future_time += self.transition_system['actions']["weight"]
                        self.pose_history.append((self.pose, self.next_interval+last_time, future_step, future_time))
                        return
            
    def simulate(self):
        
        self.plan_index = 0
        self.t = rospy.Time.now()
        self.next_interval = 10
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise ValueError("pygame shutdown")

                # Clear the screen
                self.world.screen.fill(WHITE)
                self.world.background()
                # Draw the grid
                for row in range(self.world.grid_size):
                    for col in range(self.world.grid_size):
                        pygame.draw.rect(self.world.screen, BLACK, (col * self.world.cell_size, row * self.world.cell_size, self.world.cell_size, self.world.cell_size), 1)
                
                # Update to next action
                if (rospy.Time.now() - self.t).to_sec() >= self.next_interval/10:
                    self.next_move()
                
                # Draw the moving agent
                text = pygame.font.SysFont('timesnewroman', 10).render(str(self.mode), True, BLACK, WHITE)
                text_ = text.get_rect()
                text_.center = (int(self.pose[0] * self.world.cell_size + self.world.cell_size/2), \
                    self.world.height - int(self.pose[1] * self.world.cell_size + self.world.cell_size/2))
                
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
                        
                pygame.draw.circle(self.world.screen, self.mode.value, (int(self.pose[0] * self.world.cell_size + self.world.cell_size/2), \
                    self.world.height - int(self.pose[1] * self.world.cell_size + self.world.cell_size/2)), self.world.cell_size // 2)
                self.world.screen.blit(text, text_)
                
                
                pygame_surface = pygame.display.get_surface()
                pygame_pixels = pygame.surfarray.array3d(pygame_surface)
                image = np.flipud(pygame_pixels)

                # Convert to BGR format (required by OpenCV)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Write the frame to the video file
                self.world.output_video.write(image)
                
                # Update the display
                pygame.display.flip()
                

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
                
                rate.sleep()    
            except KeyboardInterrupt:
                print(self.pose_history)
                csv_file_name = "example.csv"
                with open(csv_file_name, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for row in self.pose_history:
                        writer.writerow(row)

#==============================
#             Main
#==============================
if __name__ == '__main__':
    # Quit Pygame
    pygame.init()
    env = GridWorld()
    rospy.init_node('ltl_drone', anonymous=False, disable_signals=True)
    try:
        ltl_drone = LTLControllerDrone(env)
        rospy.spin()
    # except KeyboardInterrupt:
    #     print(ltl_drone.pose_history)
    #     env.output_video.release()
    #     pygame.quit()
    #     sys.exit(0)
        
    except ValueError as e:
        rospy.logerr("LTL drone node: %s" %(e))
        env.output_video.release()
        pygame.quit()
        sys.exit(0)
    