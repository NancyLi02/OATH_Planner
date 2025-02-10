import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import uuid
from ltl_automaton_msgs.msg import TaskRequest, TaskAssignment, PositionRequest, CurrentPosition
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
                    print(f"\n****************Assigning task for Robot {robot_index + 1}:****************")
                    task_index = self.select_task(scores_list, y, robot_index)
                    if task_index is not None:
                        x, assigned_tasks = self.conflict_resolve(task_index, assigned_tasks, x)
                        x[robot_index][task_index] = 1
                        assigned_tasks.append((robot_index, task_index))
                    else:
                        print(f"Robot {robot_index + 1} could not be assigned any task.")

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
        self.declare_parameter('robot1_initial_state', 'c0_r0')
        self.declare_parameter('robot2_initial_state', 'c4_r3')

        # Retrieve initial states from parameters
        self.robot1_initial_state = self.get_parameter('robot1_initial_state').value
        self.robot2_initial_state = self.get_parameter('robot2_initial_state').value
        self.robot_init_state = [self.robot1_initial_state, self.robot2_initial_state]        

        self.get_logger().info(f'Robot 1 initial state: {self.robot1_initial_state}')
        self.get_logger().info(f'Robot 2 initial state: {self.robot2_initial_state}')
        
        # Initialize CBAA algorithm and robot states
        self.cbaa_algorithm = CBAA()

        # Task positions hard coded for now
        self.task_positions = ['c5_r0', 'c5_r5', 'c0_r3', 'c0_r5']
        self.valid_tasks = [1, 1, 1, 1]
        # Convert positions from format like 'c5_r0' to coordinates (x, y)
        self.task_pos = [
            (int(pos.split('_')[0][1:]), int(pos.split('_')[1][1:])) for pos in self.task_positions
            ]
        self.robot_pos = [
            (int(pos.split('_')[0][1:]), int(pos.split('_')[1][1:])) for pos in self.robot_init_state
            ]
        self.positions = [None, None]  # To store positions for robot_1 and robot_2
        self.task_assignment_topic = ''

        # Publishers to send task assignments
        self.task_assignment_publisher1 = self.create_publisher(TaskAssignment, 'robot1/task_assignment', 10)
        self.task_assignment_publisher2 = self.create_publisher(TaskAssignment, 'robot2/task_assignment', 10)
        self.position_request_publisher1 = self.create_publisher(PositionRequest, 'robot1/position_request', 10)
        self.position_request_publisher2 = self.create_publisher(PositionRequest, 'robot2/position_request', 10)

        # Subscriber to receive task assignment requests
        self.task_request_subscriber = self.create_subscription(
            TaskRequest,
            'task_assignment_request',
            self.pub_new_taskassignment,
            10
        )
        self.position_receive_subscriber = self.create_subscription(
            CurrentPosition,
            'current_position',
            self.position_receive_callback,
            10
        )

        self.get_logger().info('TaskAssignNode has been started.')

        time.sleep(1)
        # Publish Initial Task Assignments
        self.pub_initial_tasks(self.robot_pos, self.task_pos)
        self.get_logger().info('Initial tasks have been assigned.')
    
    def position_receive_callback(self, msg):
        # Update robots’ current position
        self.positions[msg.robot_id] = msg.position
        self.get_logger().info(
            f'Received position from {msg.robot_id}: {msg.position}'
        )

    def pub_initial_tasks(self, robot_pos, task_pos):
        # Calculate scores for each robot-task pair
        scores_list = self.calculate_score(robot_pos, self.valid_tasks, task_pos)

        # Perform task auction using CBAA algorithm
        assigned_tasks = self.cbaa_algorithm.initial_task_assignment(scores_list)

        # Publish initial task assignment
        self.publish_task_assignment(assigned_tasks)

    def pub_new_taskassignment(self, msg):
        # Publish position request to robots
        request_id = str(uuid.uuid4())
        position_request_msg = PositionRequest()
        position_request_msg.request_id = request_id
        self.position_request_publisher1.publish(position_request_msg)
        self.position_request_publisher2.publish(position_request_msg)
        self.get_logger().info(f'Sent position request with ID: {request_id}')

        # Extract robot and task information from message
        robot_id = msg.robot_id
        task_id = msg.task_id
        task_status = msg.task_status

        # Update valid tasks if task is completed
        if task_status == 1:
            if 0 <= task_id < len(self.valid_tasks):
                self.valid_tasks[task_id] = 0  # Mark task as invalid

        self.get_logger().info(f'Received task assignment request from Robot {robot_id}.')

        # Check if both robots have reported their positions
        if all(position is not None for position in self.positions):
            robot_position = self.positions  # Both positions are available
            task_list = self.task_pos

            # Calculate task scores and assign tasks
            score_list = self.calculate_score(robot_position, self.valid_tasks, task_list)
            assigned_tasks = self.cbaa_algorithm.initial_task_assignment(score_list)

            # Publish task assignment
            self.publish_task_assignment(assigned_tasks)
            self.positions = [None, None]
    
    def publish_task_assignment(self, assigned_tasks):
        task_assignment_msg = TaskAssignment()

        for robot_index, task_index in assigned_tasks:
            task_index = int(task_index)
            if robot_index == 0:
                task_assignment_msg.robot_1_task = task_index + 1
            elif robot_index == 1:
                task_assignment_msg.robot_2_task = task_index + 1

        self.get_logger().info(
            f'Publishing task assignments: Robot 1: {task_assignment_msg.robot_1_task}, Robot 2: {task_assignment_msg.robot_2_task}'
            )
        
        self.task_assignment_publisher1.publish(task_assignment_msg)
        self.task_assignment_publisher2.publish(task_assignment_msg)

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
