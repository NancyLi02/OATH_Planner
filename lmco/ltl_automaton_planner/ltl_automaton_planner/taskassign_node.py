import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import uuid
from ltl_automaton_msgs.msg import TaskRequest, TaskAssignment, TaskReAssignment, PositionRequest, CurrentPosition, ScoreRequest, ScoreList
import time

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

        # Declare parameters for initial robot states
        self.declare_parameter('robot1_initial_state', [0.0, 0.0])
        self.declare_parameter('robot2_initial_state', [0.0, 0.0])
        self.declare_parameter('score_scheme','')

        # Retrieve initial states from parameters
        self.robot1_initial_state = self.get_parameter('robot1_initial_state').value
        self.robot2_initial_state = self.get_parameter('robot2_initial_state').value
        self.robot_init_state = [self.robot1_initial_state, self.robot2_initial_state] 
        self.score_scheme  = self.get_parameter('score_scheme').get_parameter_value().string_value     

        self.get_logger().info(f'Robot 1 initial state: {self.robot1_initial_state}')
        self.get_logger().info(f'Robot 2 initial state: {self.robot2_initial_state}')
        self.get_logger().info(f'Score scheme for CBAA: {self.score_scheme}')
        
        # Initialize CBAA algorithm and robot states
        self.cbaa_algorithm = CBAA()
        self.assigned_tasks = {}
        self.previous_assigned_tasks = {}
        self.unloaded_robots = {
            "robot_1": 180,
            "robot_2": 182
        }
        self.busy_robots = {}
        self.first_call = True

        # Task positions hard coded for now
        self.task_positions = [(5.5, 0.3), (5.5, 2.7), (0.5, 1.7), (0.5, 2.7)]
        self.valid_tasks = [1, 1, 1, 1, 1, 1]
        # Convert positions from format like 'c5_r0' to coordinates (x, y)
        self.task_pos = [
            (pos[0], pos[1]) for pos in self.task_positions
        ]
        self.positions = {}
        self.pose_index_list = {}
        self.score_list = {}

        # Define robot list
        robots = ['robot1', 'robot2']

        # Dictionaries to store publishers
        # self.task_assignment_publishers = {}
        self.task_reassignment_publishers = {}
        self.position_request_publishers = {}
        self.update_pose_pub = {}
        self.score_request_publishers = {}

        # Loop to create publishers
        for robot in robots:
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
        for robot in robots:
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
        if msg.robot_id not in self.pose_index_list:
            self.pose_index_list[msg.robot_id] = {
                "pose_index": msg.pose_index,
                "current_state": msg.current_state
            }
            self.get_logger().info(
                f'Received pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
            )

        if "robot_1" not in self.pose_index_list or "robot_2" not in self.pose_index_list:
            self.pose_index_list[msg.robot_id]["pose_index"] = msg.pose_index
            self.pose_index_list[msg.robot_id]["current_state"] = msg.current_state

            self.get_logger().info(
                f'Received pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
            )

        if "robot_1" in self.pose_index_list and "robot_2" in self.pose_index_list:
            self.get_logger().info("Both robot_1 and robot_2 positions collected, assigning new task...")
            self.new_task_assign(self.pose_index_list)
            self.pose_index_list = {}
    
    def update_pose_callback(self, msg):
        if msg.robot_id not in self.pose_index_list:
            self.pose_index_list[msg.robot_id] = {
                "pose_index": msg.pose_index,
                "current_state": msg.current_state
            }
            self.get_logger().info(
                f'Received update pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
            )

        if "robot_1" not in self.pose_index_list or "robot_2" not in self.pose_index_list:
            self.pose_index_list[msg.robot_id]["pose_index"] = msg.pose_index
            self.pose_index_list[msg.robot_id]["current_state"] = msg.current_state

            self.get_logger().info(
                f'Received update pose index from {msg.robot_id}: {msg.pose_index}, state: {msg.current_state}'
            )

        if "robot_1" in self.pose_index_list and "robot_2" in self.pose_index_list:
            self.get_logger().info("Both robot_1 and robot_2 update pose index collected, publishing new task...")
            self.check_task_assign(self.pose_index_list)
            self.publish_reassignment(self.unloaded_robots_update)
            self.pose_index_list = {}


    def pub_score_request(self, positions):
        if "robot_1" in positions and "robot_2" in positions:
            # Create ScoreRequest messages for both robots
            score_request_msg1 = ScoreRequest()
            score_request_msg1.position = positions["robot_1"]["position"]

            score_request_msg2 = ScoreRequest()
            score_request_msg2.position = positions["robot_2"]["position"]

            # Publish messages
            self.score_request_pub1.publish(score_request_msg1)
            self.score_request_pub2.publish(score_request_msg2)

            self.get_logger().info("Published score request for robot_1 and robot_2.")
        else:
            self.get_logger().warn("Missing positions for one or both robots. Score request not published.")

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


    def new_task_assign(self, pose_index_list):
        # Filter out all robots whose state is not 'unloaded'
        self.unloaded_robots = {}
        self.busy_robots = {}

        if self.score_scheme == 'dstar':
            self.get_logger().info('Using dstar as score scheme.........')
            for robot_id, data in pose_index_list.items():
                if data["current_state"] == "unloaded":
                    self.unloaded_robots[robot_id] = data["pose_index"]
                    
                    score_request_msg = ScoreRequest()
                    score_request_msg.pose_index = self.unloaded_robots[robot_id]
                    
                    if robot_id == 'robot_1':
                        self.score_request_publishers['robot1'].publish(score_request_msg)
                    elif robot_id == 'robot_2':
                        self.score_request_publishers['robot2'].publish(score_request_msg)
                    
                    self.get_logger().info(f"Published score request for {robot_id}")
                else:
                    self.busy_robots[robot_id] = data["pose_index"]

        # if self.score_scheme == 'manhattan':
        #     for robot_id, data in pose_index_list.items():
        #         if data["current_state"] == "unloaded":
        #             self.unloaded_robots[robot_id] = data["pose_index"]
        #         else:
        #             self.busy_robots[robot_id] = data["pose_index"]
        
        #     self.valid_tasks = self.construct_valid_tasks(self.busy_robots)
            
        #     # If no available robots, return immediately
        #     if not self.unloaded_robots:
        #         self.get_logger().info("No available robots for task assignment.")
        #         return
            
        #     # Sort unloaded_robots based on robot_id numerically
        #     sorted_unloaded_robots = dict(sorted(self.unloaded_robots.items(), key=lambda x: int(x[0].split('_')[-1])))
            
        #     # Extract positions of sorted unloaded robots and create a mapping
        #     robot_positions = list(sorted_unloaded_robots.values())
        #     robot_id_mapping = {robot_id: index for index, robot_id in enumerate(sorted_unloaded_robots.keys())}
        #     reverse_robot_id_mapping = {index: int(robot_id.split('_')[-1]) for robot_id, index in robot_id_mapping.items()}
            
        #     # Calculate task scores for each unloaded robot
        #     scores_list = self.calculate_score(robot_positions, self.valid_tasks, self.task_pos)
            
        #     # Perform task assignment
        #     assigned_tasks = self.cbaa_algorithm.initial_task_assignment(scores_list)
            
        #     # Map numerical indices back to expected format
        #     self.previous_assigned_tasks = self.assigned_tasks
        #     self.assigned_tasks = []
            
        #     for robot_index, task_index in assigned_tasks:
        #         assigned_robot_number = reverse_robot_id_mapping[robot_index] - 1  # Convert robot_index to expected format
        #         self.assigned_tasks.append((assigned_robot_number, int(task_index)))  # Ensure task_index is an integer
            
        #     self.get_logger().info(f"Final task assignments: {self.assigned_tasks}")
            
        #     # Publish task assignment
        #     self.publish_reassignment(self.unloaded_robots)


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
            
            # Record previous task assignments
            self.previous_assigned_tasks = self.assigned_tasks
            self.assigned_tasks = []

            # Sort unloaded_robots based on robot_id
            sorted_unloaded_robots = dict(sorted(self.unloaded_robots.items(), key=lambda x: int(x[0].split('_')[-1])))
            
            # Create robot_id mappings
            robot_id_mapping = {robot_id: index for index, robot_id in enumerate(sorted_unloaded_robots.keys())}
            reverse_robot_id_mapping = {index: int(robot_id.split('_')[-1]) for robot_id, index in robot_id_mapping.items()}
            

            for robot_index, task_index in assigned_tasks:
                assigned_robot_number = reverse_robot_id_mapping[robot_index] - 1  # Convert robot_index to expected format
                self.assigned_tasks.append((assigned_robot_number, int(task_index)))  # Ensure task_index is an integer

            self.get_logger().info(f"Final task assignments: {self.assigned_tasks}")
            
            # Publish task assignment
            if not self.first_call:
                self.update_pose()
            else:
                self.publish_reassignment(self.unloaded_robots)
                self.first_call = False
                self.score_list = {}
                self.positions = {}

    def update_pose(self):
        request_id = str(uuid.uuid4())
        position_request_msg = PositionRequest()
        position_request_msg.request_id = request_id
        self.update_pose_pub['robot1'].publish(position_request_msg)
        self.update_pose_pub['robot2'].publish(position_request_msg)
        self.get_logger().info(f'Sent update position request with ID: {request_id}')

    # def pub_initial_tasks(self, robot_pos, task_pos):
    #     # Calculate scores for each robot-task pair
    #     scores_list = self.calculate_score(robot_pos, self.valid_tasks, task_pos)

    #     # Perform task auction using CBAA algorithm
    #     self.assigned_tasks = self.cbaa_algorithm.initial_task_assignment(scores_list)

    #     # Publish initial task assignment
    #     self.publish_task_assignment(self.assigned_tasks)

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
        self.position_request_publishers['robot1'].publish(position_request_msg)
        self.position_request_publishers['robot2'].publish(position_request_msg)
        self.get_logger().info(f'Sent position request with ID: {request_id}')

    
    # def publish_task_assignment(self, assigned_tasks):
    #     task_assignment_msg = TaskAssignment()

    #     # Assign tasks to robots based on the assignment results
    #     for robot_index, task_index in assigned_tasks:
    #         task_index = int(task_index)
    #         if robot_index == 0:
    #             task_assignment_msg.robot_1_task = task_index + 1
    #         elif robot_index == 1:
    #             task_assignment_msg.robot_2_task = task_index + 1

    #     # Log the task assignment results
    #     self.get_logger().info(
    #         f'Publishing task assignments: Robot 1: {task_assignment_msg.robot_1_task}, Robot 2: {task_assignment_msg.robot_2_task}'
    #     )

    #     # Publish the task assignment message to both publishers
    #     self.task_assignment_publishers['robot1'].publish(task_assignment_msg)
    #     self.task_assignment_publishers['robot2'].publish(task_assignment_msg)



    def publish_reassignment(self, unloded_robot):
        task_assignment_msg = TaskReAssignment()

        publish_to_robot_1 = False
        publish_to_robot_2 = False

        # Assign tasks and include robot positions
        for robot_index, task_index in self.assigned_tasks:
            task_index = int(task_index)
            
            if (robot_index == 0 and "robot_1" in unloded_robot):
                # Check if the previous assignment for robot_1 was the same, if so, do not change publish_to_robot_1
                if not any(prev_robot == 0 and prev_task == task_index for prev_robot, prev_task in self.previous_assigned_tasks):
                    publish_to_robot_1 = True
                    task_assignment_msg.robot_1_task = task_index + 1 if task_index != -1 else -1
                    task_assignment_msg.robot_1_pos = unloded_robot['robot_1']  # Add robot 1 position

            elif (robot_index == 1 and "robot_2" in unloded_robot):
                # Check if the previous assignment for robot_2 was the same, if so, do not change publish_to_robot_2
                if not any(prev_robot == 1 and prev_task == task_index for prev_robot, prev_task in self.previous_assigned_tasks):
                    publish_to_robot_2 = True
                    task_assignment_msg.robot_2_task = task_index + 1
                    task_assignment_msg.robot_2_pos = unloded_robot['robot_2']  # Add robot 2 position

        # Log the task assignment results
        self.get_logger().info(
            f'Publishing new task assignments {self.assigned_tasks}. ' 
            f'Publish to robot1 {publish_to_robot_1}, '
            f'Publish to robot2 {publish_to_robot_2}. '
        )

        # Publish the task assignment message to the respective publishers if applicable
        if publish_to_robot_1:
            self.task_reassignment_publishers['robot1'].publish(task_assignment_msg)
        if publish_to_robot_2:
            self.task_reassignment_publishers['robot2'].publish(task_assignment_msg)
        
            
    def calculate_score(self, robot_position, valid_tasks, task_list):
        scores_list = []
        for i, robot in enumerate(robot_position):
            robot_scores = []
            for j, task in enumerate(task_list):
                if valid_tasks[j] == 1:
                    manhattan_distance = np.abs(robot[0] - task[0]) + np.abs(robot[1] - task[1])
                    score = 15 - manhattan_distance
                else:
                    score = 0
                robot_scores.append(score)
            scores_list.append(robot_scores)
        return np.array(scores_list)

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
