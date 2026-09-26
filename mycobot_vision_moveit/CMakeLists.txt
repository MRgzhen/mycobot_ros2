cmake_minimum_required(VERSION 3.20)
project(mycobot_vision_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(move_to_point src/move_to_point.cpp)
target_compile_features(move_to_point PUBLIC cxx_std_17)
target_link_libraries(move_to_point
  ${geometry_msgs_TARGETS}
  ${moveit_ros_planning_interface_TARGETS}
  ${rclcpp_TARGETS}
)

install(TARGETS move_to_point DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
