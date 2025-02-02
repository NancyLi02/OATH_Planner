from cbaa import initial_task_assignment
import numpy as np
import matplotlib.pyplot as plt


agent_positions = [
    np.array([10, 15]), 
    np.array([45, 4]),
    np.array([2, 35]),
    np.array([3,30]),
    np.array([10,20])
]

targer_num = 10
target_list = np.random.uniform(0, 50, size=(targer_num, 2))

scores_list = []
for i, robot in enumerate(agent_positions):
    robot_scores = []
    for target in target_list:
        manhattan_distance = np.abs(robot[0] - target[0]) + np.abs(robot[1] - target[1])
        score = (100-manhattan_distance)
        robot_scores.append(score)
    scores_list.append(robot_scores)

scores_list = np.array(scores_list)

# Initial task assignment
assigned_tasks = initial_task_assignment(scores_list)

