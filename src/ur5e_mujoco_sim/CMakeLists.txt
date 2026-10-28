cmake_minimum_required(VERSION 3.8)
project(ur5e_mujoco_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(ur5e_move_group native/move_group_main.cpp)
target_compile_features(ur5e_move_group PUBLIC cxx_std_17)
ament_target_dependencies(ur5e_move_group moveit_core moveit_ros_planning pluginlib rclcpp)
target_link_libraries(ur5e_move_group
  moveit_ros_move_group::moveit_move_group_capabilities_base)
install(TARGETS ur5e_move_group RUNTIME DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(project_regressions test/test_project_regressions.py)
endif()

install(DIRECTORY
  models
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS src/pick_place.py DESTINATION lib/${PROJECT_NAME} RENAME pick_place)
install(PROGRAMS src/plot_pid_response.py DESTINATION lib/${PROJECT_NAME} RENAME plot_pid_response)
install(PROGRAMS src/plot_pid_response_live.py DESTINATION lib/${PROJECT_NAME} RENAME plot_pid_response_live)
install(PROGRAMS src/pid_step_response.py DESTINATION lib/${PROJECT_NAME} RENAME pid_step_response)
install(PROGRAMS src/vision_pick_place.py DESTINATION lib/${PROJECT_NAME} RENAME vision_pick_place)
install(PROGRAMS src/vision_camera_preview.py DESTINATION lib/${PROJECT_NAME} RENAME vision_camera_preview)
install(DIRECTORY
  src/pick_place_lib
  src/trajectory_lib
  src/vision_lib
  src/pid_lib
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY src/pid_lib DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
ament_package()
