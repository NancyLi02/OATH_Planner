# TaskRequest.msg
# This message defines a task request structure with a robot index and a valid task list

# The robot's unique identifier
int32 robot_id

# # The robot's position represented as an index in the grid
# int32 pose_index

# Whether the robot finished the task, 1=finished, 0=failed
int32 task_status
