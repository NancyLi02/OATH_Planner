# CurrentPosition.msg
# This message defines the current position structure of a robot

# The robot's unique identifier
string robot_id

# The current position of the robot, represented as a 2D coordinate (x, y)
int32[] position

# Current situation 'loaded' or 'unloaded'
string current_state