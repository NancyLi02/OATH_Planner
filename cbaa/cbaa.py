import numpy as np
import matplotlib.pyplot as plt
import time

def select_task(scores_list, y, robot_index):
    """
    Select the best task for the given robot based on scores and constraints.
    """
    robot_scores = scores_list[robot_index]
    valid_task = [(1 if score > y_value else 0) for score, y_value in zip(robot_scores, y)]

    # Check for valid tasks
    if sum(valid_task) > 0:
        valid_indices = [i for i, valid in enumerate(valid_task) if valid == 1]
        best_task_index = max(valid_indices, key=lambda idx: robot_scores[idx])
        best_task_score = robot_scores[best_task_index]

        # Update the y list with the score of the selected task
        y[best_task_index] = best_task_score

        print(f"Robot {robot_index + 1}: Selected task index {best_task_index} with score {best_task_score:.2f}")
        return best_task_index
    else:  # No valid task found
        print(f"Robot {robot_index + 1}: No valid task available.")
        return None

def conflict_resolve(task_index, assigned_tasks, x):
    """
    Resolve conflicts if the selected task is already assigned to another robot.
    """
    for robot, assigned_task in assigned_tasks:
        if assigned_task == task_index:
            print(f"Conflict detected for task {task_index}. Removing previous assignment for Robot {robot + 1}.")
            # Remove the task assignment for the previous robot
            x[robot][task_index] = 0
            # Remove the conflict entry from the assigned tasks
            assigned_tasks.remove((robot, assigned_task))
            break
    return x, assigned_tasks

def initial_task_assignment(scores_list):
    """
    Assign tasks to robots based on their scores and constraints, resolving conflicts as needed.
    """
    task_count = len(scores_list[0])
    x = [[0] * task_count for _ in range(len(scores_list))]  # Task assignment matrix
    y = [0] * task_count  # Bidding threshold for each task
    assigned_tasks = []  # Keep track of assigned tasks

    # Assign tasks until each robot has at least one task
    while any(sum(row) == 0 for row in x):
        for robot_index in range(len(scores_list)):
            # Skip if the current robot already has a task assigned
            if sum(x[robot_index]) == 0:
                print(f"\n****************Assigning task for Robot {robot_index + 1}:****************")

                # Select the best task for the robot
                task_index = select_task(scores_list, y, robot_index)

                if task_index is not None:
                    # Resolve conflicts if the task is already assigned
                    x, assigned_tasks = conflict_resolve(task_index, assigned_tasks, x)

                    # Assign the task to the current robot
                    x[robot_index][task_index] = 1
                    assigned_tasks.append((robot_index, task_index))
                else:
                    print(f"Robot {robot_index + 1} could not be assigned any task.")

    # Display final results
    print("\nFinal y list:", y)
    print("x list (task assignment status for each robot):")
    for robot_index, task_assignments in enumerate(x):
        print(f"Robot {robot_index + 1}: {task_assignments}")

    return assigned_tasks  # Return the task assignment results
