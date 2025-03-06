# ShowPosition.msg
# This message defines the current position structure of a robot

# The robot's unique identifier
string robot_id

# The current pose
float64[] pose

# Current situation 'loaded' or 'unloaded'
string mode