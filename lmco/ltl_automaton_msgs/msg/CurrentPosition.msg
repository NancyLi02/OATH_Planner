# CurrentPosition.msg
# This message defines the current position structure of a robot

# The robot's unique identifier
string robot_id

# The current pose_index
int32 pose_index

# Current situation 'loaded' or 'unloaded'
string current_state