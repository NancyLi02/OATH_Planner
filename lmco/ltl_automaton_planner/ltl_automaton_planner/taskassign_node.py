import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from ltl_automaton_msgs.msg import TaskRequest
import numpy as np
import matplotlib.pyplot as plt
import random

#=================================================================
#  Interfaces between TaskAssignNode and other nodes
#                       -----------------
# Created by Nan Li
# The node is responsible for receiving task assignment requests,
# calculating task scores for each agent, and assigning tasks
# based on the CBAA (Consensus-Based Auction Algorithm).
# It communicates with the planner and benchmark nodes.
#=================================================================

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

class TaskAssignNode(Node):
    def __init__(self):
        super().__init__('taskassign_node')
        
        self.cbaa_algorithm = CBAA()
        # Publisher to send task assignments to planner_node
        self.task_assignment_publisher = self.create_publisher(String, 'task_assignment', 10) # 需要修改消息类型！！！！

        # Subscriber to receive task assignment requests from benchmark_node
        self.task_request_subscriber = self.create_subscription(
            TaskRequest,
            'task_assignment_request',
            self.task_request_callback,
            10
        )
        
        self.get_logger().info('TaskAssignNode has been started.')

    def task_request_callback(self, msg):
        # robot_position = msg.robot_position  # `robot_position` is defined in TaskRequest
        # valid_tasks = msg.valid_tasks        # `valid_tasks` is defined in TaskRequest

        # Test robot_position and valid_tasks
        robot_position = [np.array([10, 15]), np.array([45, 4]), np.array([2, 35]), np.array([3,30]), np.array([10,20])]
        valid_tasks = [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        task_num = len(valid_tasks)
        task_list = np.random.uniform(0, 50, size=(task_num, 2))
        self.get_logger().info(f'Received task assignment request. Start task assignment progress')

        score_list = self.calculate_score(robot_position, valid_tasks, task_list)
        task_assignment = self.cbaa_algorithm.initial_task_assignment(score_list)
        self.get_logger().info(f'Task assignment completed: {task_assignment}')

        assignment_msg = String()
        assignment_msg.data = str(task_assignment)
        self.task_assignment_publisher.publish(assignment_msg)

    def calculate_score(robot_position, valid_tasks, task_list):
        scores_list = []

        for i, robot in enumerate(robot_position):
            robot_scores = []
            for j, task in enumerate(task_list):
                if valid_tasks[j] == 1:
                    # Calculate the Manhattan distance between the robot and the task
                    manhattan_distance = np.abs(robot[0] - task[0]) + np.abs(robot[1] - task[1])
                    
                    # Calculate the score as 100 - Manhattan distance
                    score = 100 - manhattan_distance
                else:
                    # If the task is not valid, the score is 0
                    score = 0
                
                robot_scores.append(score)
            
            scores_list.append(robot_scores)

        return np.array(scores_list)
    
#==============================
#             Main
#==============================

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
