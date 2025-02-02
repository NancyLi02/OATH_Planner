# TaskRequest.msg
# This message defines a task request structure with a robot index and a valid task list

# The robot's unique identifier
int32 robot_id

# A list of valid task IDs assigned to the robot
int32[] valid_tasks

