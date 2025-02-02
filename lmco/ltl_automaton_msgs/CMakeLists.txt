cmake_minimum_required(VERSION 3.8)
project(ltl_automaton_msgs)

# Default to C++14 if available
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages in the 'msg' folder
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TransitionSystemState.msg"
  "msg/TransitionSystemStateStamped.msg"
  "msg/LTLPlan.msg"
  "msg/LTLState.msg"
  "msg/LTLStateArray.msg"
  "msg/LTLStateRuns.msg"
  "msg/RelayRequest.msg"
  "msg/RelayResponse.msg"
  
  # Generate services in the 'srv' folder
  "srv/TrapCheck.srv"
  "srv/TaskPlanning.srv"
  "srv/TaskReplanningModify.srv"
  "srv/TaskReplanningDelete.srv"
  "srv/TaskReplanningRelabel.srv"
  
  DEPENDENCIES std_msgs builtin_interfaces
)

# # Install includes, launch files, and any other package files
# install(DIRECTORY launch config
#   DESTINATION share/${PROJECT_NAME}/
# )

ament_export_dependencies(rosidl_default_runtime rclcpp std_msgs)
ament_package()