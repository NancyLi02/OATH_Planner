# TaskRequest.msg
# This message defines a task request structure with a robot index and a valid task list

# The robot's unique identifier
int32 robot_id

# The robot's position in (x, y) format
int32 position_x
int32 position_y

# Whether the robot finished the task, 1=finished, 0=failed
int32 task_status
