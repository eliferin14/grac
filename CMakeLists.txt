cmake_minimum_required(VERSION 3.0.2)
project(gesture_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
  actionlib_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  draw.msg
  plot.msg
  timestamps.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gesture_control
  CATKIN_DEPENDS message_runtime geometry_msgs sensor_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


install(
  DIRECTORY src/gesture_control
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(
  DIRECTORY src/gesture_utils
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
