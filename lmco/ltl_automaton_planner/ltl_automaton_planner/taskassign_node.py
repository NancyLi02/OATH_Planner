import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import uuid
from ltl_automaton_msgs.msg import TaskRequest, TaskAssignment, TaskReAssignment, PositionRequest, CurrentPosition, ScoreRequest, ScoreList
import time
import copy

#=================================================================
#  Interfaces between TaskAssignNode and other nodes
#                       -----------------
# Created by Nan Li
# This node is responsible for receiving task assignment requests,
# calculating task scores for each agent, and assigning tasks
# based on the CBAA (Consensus-Based Auction Algorithm).
#=================================================================

# Consensus-Based Auction Algorithm
class CBAA:
    def __init__(self):
        pass

    def select_task(self, scores_list, y, robot_index):
        """
        Select the best task for the given robot based on scores and constraints.
        """
        robot_scores = scores_list[robot_index]
        valid_task = [(1 if score > y_value else 0) for score, y_value in zip(robot_scores, y)]

        if sum(valid_task) > 0:
            valid_indices = [i for i, valid in enumerate(valid_task) if valid == 1]
            best_task_index = max(valid_indices, key=lambda idx: robot_scores[idx])
            best_task_score = robot_scores[best_task_index]
            y[best_task_index] = best_task_score  # Update bid score list
            print(f"Robot {robot_index + 1}: Selected task index {best_task_index} with score {best_task_score:.2f}")
            return best_task_index
        else:
            print(f"Robot {robot_index + 1}: No valid task available.")
            return None

    def conflict_resolve(self, task_index, assigned_tasks, x):
        for robot, assigned_task in assigned_tasks:
            if assigned_task == task_index:
                print(f"Conflict detected for task {task_index}. Removing previous assignment for Robot {robot + 1}.")
                x[robot][task_index] = 0
                assigned_tasks.remove((robot, assigned_task))
                break
        return x, assigned_tasks

    def initial_task_assignment(self, scores_list):
        task_count = len(scores_list[0])
        x = [[0] * task_count for _ in range(len(scores_list))]
        y = [0] * task_count
        assigned_tasks = []

        while any(sum(row) == 0 for row in x):
            for robot_index in range(len(scores_list)):
                if sum(x[robot_index]) == 0:
                    #print(f"\n****************Assigning task for Robot {robot_index + 1}:****************")
                    task_index = self.select_task(scores_list, y, robot_index)
                    if task_index is not None:
                        x, assigned_tasks = self.conflict_resolve(task_index, assigned_tasks, x)
                        x[robot_index][task_index] = 1
                        assigned_tasks.append((robot_index, task_index))
                    else:
                        #print(f"Robot {robot_index + 1} could not be assigned any task.")
                        return

        print("\nFinal y list:", y)
        print("x list (task assignment status for each robot):")
        for robot_index, task_assignments in enumerate(x):
            print(f"Robot {robot_index + 1}: {task_assignments}")

        return assigned_tasks


# Task Assignment Node
class TaskAssignNode(Node):
    def __init__(self):
        super().__init__('taskassign_node')

        self.declare_parameter('robot_count', 4)
        self.robot_count = self.get_parameter('robot_count').value
        
        self.robot_init_poses = {}
        for i in range(1, self.robot_count + 1):
            param_name = f'robot{i}_init_pose'
            self.declare_parameter(param_name, 0)
            self.robot_init_poses[f'robot_{i}'] = self.get_parameter(param_name).value
        
        self.declare_parameter('score_scheme', '')
        self.score_scheme = self.get_parameter('score_scheme').get_parameter_value().string_value
        
        for robot, init_pose in self.robot_init_poses.items():
            self.get_logger().info(f'{robot} initial state: {init_pose}')
        self.get_logger().info(f'Score scheme for CBAA: {self.score_scheme}')
        
        self.cbaa_algorithm = CBAA()
        self.assigned_tasks = []
        self.previous_assigned_tasks = []
        self.unloaded_robots = {robot: pose for robot, pose in self.robot_init_poses.items()}
        self.busy_robots = {}
        self.first_call = True

        self.declare_parameter('task_count', 10)
        task_count = self.get_parameter('task_count').value
        self.valid_tasks = [1] * task_count

        self.positions = {}
        self.pose_index_list = {}
        self.score_list = {}

        # Define robot list
        self.robots = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        self.robots_ = [f'robot_{i}' for i in range(1, self.robot_count + 1)]

        # Dictionaries to store publishers
        # self.task_assignment_publishers = {}
        self.task_reassignment_publishers = {}
        self.position_request_publishers = {}
        self.update_pose_pub = {}
        self.score_request_publishers = {}

        # Loop to create publishers
        for robot in self.robots:
            # self.task_assignment_publishers[robot] = self.create_publisher(
            #     TaskAssignment, f'{robot}/task_assignment', 10)
            
            self.task_reassignment_publishers[robot] = self.create_publisher(
                TaskReAssignment, f'{robot}/task_reassignment', 10)
            
            self.position_request_publishers[robot] = self.create_publisher(
                PositionRequest, f'{robot}/position_request', 10)
            
            self.update_pose_pub[robot] = self.create_publisher(
                PositionRequest, f'{robot}/update_pose_request', 10)
            
            self.score_request_publishers[robot] = self.create_publisher(
                ScoreRequest, f'{robot}/score_request', 10)

        # Dictionaries to store subscribers
        self.task_request_subscribers = {}
        self.position_receive_subscribers = {}
        self.update_pose_sub = {}
        self.score_list_subscribers = {}

        # Loop to create subscribers
        for robot in self.robots:
            self.task_request_subscribers[robot] = self.create_subscription(
                TaskRequest,
                f'{robot}/task_assignment_request',
                self.pub_current_pos_request,
                10
            )

            self.position_receive_subscribers[robot] = self.create_subscription(
                CurrentPosition,
                f'{robot}/current_position',
                self.position_receive_callback,
                10
            )

            self.update_pose_sub[robot] = self.create_subscription(
                CurrentPosition,
                f'{robot}/update_current_pose',
                self.update_pose_callback,
                10
            )

            self.score_list_subscribers[robot] = self.create_subscription(
                ScoreList,
                f'{robot}/score_list',
                self.score_list_receive_callback,
                10
            )

        self.get_logger().info('TaskAssignNode has been started.')

        # if self.score_scheme == 'manhattan':
        #     time.sleep(1)
        #     # Publish Initial Task Assignments
        #     self.pub_initial_tasks(self.robot_pos, self.task_pos)
        #     self.get_logger().info('Initial tasks have been assigned.')
    
    def score_list_receive_callback(self, msg): 
        robot_id = msg.robot_id  # Keep robot_id as a string, e.g., 'robot_1'
        self.get_logger().info(f'Received score list from {robot_id}')
        
        self.score_list[robot_id] = msg.score_list  # Update score list
        self.get_logger().info(f'Unloaded robots list: {self.unloaded_robots}')
        
        required_robots = set(self.unloaded_robots.keys())
        
        if required_robots.issubset(self.score_list.keys()):
            self.get_logger().info('Sufficient robot score lists collected, assigning new task...')
            self.dstar_task_assign()


    def position_receive_callback(self, msg):
        # Update or store robot position information
        self.pose_index_list[msg.robot_id] = {
            "pose_index": msg.pose_index,
            "current_state": msg.current_state
        }

        self.get_logger().info(
            f'Received pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
        )

        # Check if all required robots have reported their positions
        required_robots = {f'robot_{i}' for i in range(1, self.robot_count + 1)}
        if required_robots.issubset(self.pose_index_list.keys()):  # Verify all robots have reported
            self.get_logger().info("All robot positions collected, assigning new task...")
            self.new_task_assign(self.pose_index_list)
            self.pose_index_list.clear()  # Clear data after task assignment


    
    def update_pose_callback(self, msg):
        # Update or store robot position information
        self.pose_index_list[msg.robot_id] = {
            "pose_index": msg.pose_index,
            "current_state": msg.current_state
        }

        self.get_logger().info(
            f'Received update pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
        )
        # Check if all required robots have reported their updated pose index
        required_robots = {f'robot_{i}' for i in range(1, self.robot_count + 1)}
        if required_robots.issubset(self.pose_index_list.keys()):  # Verify all robots have updated
            self.get_logger().info("All robots update pose index collected, publishing new task...")
            self.check_task_assign(self.pose_index_list)
            self.pose_index_list.clear()  # Clear data after reassignment


    def construct_valid_tasks(self, busy_robots):
        # This function updates self.valid_tasks by setting the task index assigned to busy robots to 0
        for robot_id in busy_robots:
            robot_index = int(robot_id.split('_')[-1]) - 1  # Convert robot ID to index format
            for assigned_robot, task_index in self.assigned_tasks:
                if assigned_robot == robot_index:
                    self.valid_tasks[task_index] = 0  # Set the task as invalid
        self.get_logger().info(f"Updated valid tasks: {self.valid_tasks}")
        return self.valid_tasks

    def check_task_assign(self, pose_index_list):
        self.unloaded_robots_update = {}
        self.busy_robots_update = {}

        self.get_logger().info('Checking update robot states...')
        for robot_id, data in pose_index_list.items():
            if data["current_state"] == "unloaded":
                self.unloaded_robots_update[robot_id] = data["pose_index"]
            else:
                self.busy_robots_update[robot_id] = data["pose_index"]
        
        self.publish_reassignment(self.unloaded_robots_update)


    def new_task_assign(self, pose_index_list):
        # Filter out all robots whose state is not 'unloaded'
        self.unloaded_robots = {}
        self.busy_robots = {}

        self.get_logger().info('Using dstar as score scheme.........')
        # self.get_logger().info(f'pose_index_list is {pose_index_list}.')
        for robot_id, data in pose_index_list.items():
            if data["current_state"] == "unloaded":
                self.unloaded_robots[robot_id] = data["pose_index"]
                
                score_request_msg = ScoreRequest()
                score_request_msg.pose_index = self.unloaded_robots[robot_id]
                
                if robot_id in self.robots_:
                    # Convert robot_id from format "robot_1" to "robot1"
                    publisher_key = robot_id.replace("_", "")
                    if publisher_key in self.score_request_publishers:
                        self.score_request_publishers[publisher_key].publish(score_request_msg)
                        self.get_logger().info(f"Published score request for {robot_id}")
            else:
                self.busy_robots[robot_id] = data["pose_index"]



    def dstar_task_assign(self):
        self.valid_tasks = self.construct_valid_tasks(self.busy_robots)
        required_robots = set(self.unloaded_robots.keys())

        if required_robots.issubset(self.score_list.keys()):

            # Modify score_list based on valid_tasks
            for robot_id, scores in self.score_list.items():
                for i, valid in enumerate(self.valid_tasks):
                    if valid == 0 and i < len(scores):
                        scores[i] = 0  # Set score to 0 if task is not valid

            # Retrieve sorted score_list based on robot_id
            sorted_scores = [self.score_list.get(robot_id, []) for robot_id in sorted(self.unloaded_robots.keys(),
                                                                                    key=lambda x: int(x.split('_')[-1]))]
            
            # Perform task assignment
            assigned_tasks = self.cbaa_algorithm.initial_task_assignment(sorted_scores)  # sorted_scores is a list
            

            # Sort unloaded_robots based on robot_id
            sorted_unloaded_robots = dict(sorted(self.unloaded_robots.items(), key=lambda x: int(x[0].split('_')[-1])))
            
            # Create robot_id mappings
            robot_id_mapping = {robot_id: index for index, robot_id in enumerate(sorted_unloaded_robots.keys())}
            reverse_robot_id_mapping = {index: int(robot_id.split('_')[-1]) for robot_id, index in robot_id_mapping.items()}
            

            for robot_index, task_index in assigned_tasks:
                assigned_robot_number = reverse_robot_id_mapping[robot_index] - 1  # Convert robot_index to expected format

                # Check if the robot already exists in the list
                found = False
                for i, (r, t) in enumerate(self.assigned_tasks):
                    if r == assigned_robot_number:
                        self.assigned_tasks[i] = (assigned_robot_number, int(task_index))  # Replace existing task
                        found = True
                        break
                
                # If robot is not found, append new task
                if not found:
                    self.assigned_tasks.append((assigned_robot_number, int(task_index)))


            self.get_logger().info(f"Final task assignments: {self.assigned_tasks}")
            # Publish task assignment
            if not self.first_call:
                self.update_pose()
            else:
                self.first_call = False
                self.publish_reassignment(self.unloaded_robots)
                self.score_list = {}
                self.positions = {}

    def update_pose(self):
        request_id = str(uuid.uuid4())
        position_request_msg = PositionRequest()
        position_request_msg.request_id = request_id
        
        for robot in self.robots:
            if robot in self.update_pose_pub:
                self.update_pose_pub[robot].publish(position_request_msg)
        
        self.get_logger().info(f'Sent update position request with ID: {request_id}')
    
    def pub_current_pos_request(self, msg):
        robot_index = msg.robot_id - 1
        for assigned_robot, task_index in self.assigned_tasks:
            if assigned_robot == robot_index:
                self.valid_tasks[task_index] = 0  # Set the task as invalid
        self.get_logger().info(f"After task completed Updated valid tasks: {self.valid_tasks}")
        
        # Publish position request to robots
        request_id = str(uuid.uuid4())
        position_request_msg = PositionRequest()
        position_request_msg.request_id = request_id
        
        for robot in self.robots:
            if robot in self.position_request_publishers:
                self.position_request_publishers[robot].publish(position_request_msg)
        
        self.get_logger().info(f'Sent position request with ID: {request_id}')


    def publish_reassignment(self, unloaded_robot):
        task_assignment_msg = TaskReAssignment()
        publish_flags = {}  # Record whether to publish message for each robot

        self.get_logger().info(f'self.previous_assigned_tasks is {self.previous_assigned_tasks}')

        for robot_index, task_index in self.assigned_tasks:
            task_index = int(task_index)
            self.get_logger().info(f'robot_index is {robot_index}, task_index is {task_index}.')

            # Construct dynamic key for message fields (e.g., "robot_1", "robot_2", etc.)
            robot_msg_key = f"robot_{robot_index+1}"

            if robot_msg_key in unloaded_robot:
                self.get_logger().info(f'{robot_msg_key} in unloaded_robot')
                # Check if the previous assignment for this robot is the same as the current one
                if not any(prev_robot == robot_index and prev_task == task_index for prev_robot, prev_task in self.previous_assigned_tasks):
                    publish_flags[robot_index] = True
                    # Dynamically set task and position fields in the message
                    setattr(task_assignment_msg, f"{robot_msg_key}_task", task_index + 1 if task_index != -1 else -1)
                    setattr(task_assignment_msg, f"{robot_msg_key}_pos", unloaded_robot[robot_msg_key])
            else:
                # Update the task assignment from previous_assigned_tasks if data for this robot is not in unloaded_robot
                self.assigned_tasks = [
                    (r, prev_task) if r == robot_index else (r, t)
                    for r, t in self.assigned_tasks
                    for prev_r, prev_task in self.previous_assigned_tasks if prev_r == robot_index
                ]

        # Log the new task assignments and publishing flags for all robots
        self.get_logger().info(f'Publishing new task assignments {self.assigned_tasks}.')
        for i in range(len(self.robots)):
            flag = publish_flags.get(i, False)
            self.get_logger().info(f'Publish to {self.robots[i]} {flag}.')

        # Publish the task assignment message for robots with flag True
        for i, robot in enumerate(self.robots):
            if publish_flags.get(i, False):
                self.task_reassignment_publishers[robot].publish(task_assignment_msg)

        self.previous_assigned_tasks = copy.deepcopy(self.assigned_tasks)



def main(args=None):
    rclpy.init(args=args)
    try:
        task_assign_node = TaskAssignNode()
        rclpy.spin(task_assign_node)
    except KeyboardInterrupt:
        task_assign_node.get_logger().info("KeyboardInterrupt detected, shutting down...")
    finally:
        task_assign_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
