cmake_minimum_required(VERSION 3.4)

project(omni_slam_eval)

set(CMAKE_VERBOSE_MAKEFILE FALSE)

add_definitions("-std=c++17")
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# find catkin dependencies
set(REQ_CATKIN_PKGS
  roscpp
  cv_bridge
  sensor_msgs
  image_transport
  pcl_ros
  tf2_ros
  rosbag
)

find_package(catkin REQUIRED COMPONENTS ${REQ_CATKIN_PKGS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV 3.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

find_package(PCL 1.7 REQUIRED COMPONENTS common)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIRS})

find_package(OpenMP REQUIRED)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS ${REQ_CATKIN_PKGS}
 DEPENDS
)

include_directories(
  src
  ${catkin_INCLUDE_DIRS}
)

add_library(omni_slam_eval_lib
  src/module/tracking_module.cc
  src/module/matching_module.cc
  src/module/reconstruction_module.cc
  src/module/odometry_module.cc
  src/data/frame.cc
  src/data/feature.cc
  src/data/landmark.cc
  src/feature/tracker.cc
  src/feature/detector.cc
  src/feature/matcher.cc
  src/reconstruction/triangulator.cc
  src/odometry/pnp.cc
  src/odometry/five_point.cc
  src/optimization/bundle_adjuster.cc
  src/util/hdf_file.cc
)

add_executable(omni_slam_tracking_eval_node
    src/ros/tracking_eval.cc
    src/ros/eval_base.cc
    src/omni_slam_tracking_eval_node.cc
)

add_executable(omni_slam_matching_eval_node
    src/ros/matching_eval.cc
    src/ros/eval_base.cc
    src/omni_slam_matching_eval_node.cc
)

add_executable(omni_slam_reconstruction_eval_node
    src/ros/reconstruction_eval.cc
    src/ros/tracking_eval.cc
    src/ros/eval_base.cc
    src/omni_slam_reconstruction_eval_node.cc
)

add_executable(omni_slam_odometry_eval_node
    src/ros/odometry_eval.cc
    src/ros/tracking_eval.cc
    src/ros/eval_base.cc
    src/omni_slam_odometry_eval_node.cc
)

target_link_libraries(omni_slam_tracking_eval_node PUBLIC omni_slam_eval_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_CXX_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_link_libraries(omni_slam_matching_eval_node PUBLIC omni_slam_eval_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_CXX_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_link_libraries(omni_slam_reconstruction_eval_node PUBLIC omni_slam_eval_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_CXX_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_link_libraries(omni_slam_odometry_eval_node PUBLIC omni_slam_eval_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_CXX_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_compile_options(omni_slam_tracking_eval_node PUBLIC ${OpenMP_CXX_FLAGS})
target_compile_options(omni_slam_matching_eval_node PUBLIC ${OpenMP_CXX_FLAGS})
target_compile_options(omni_slam_reconstruction_eval_node PUBLIC ${OpenMP_CXX_FLAGS})
target_compile_options(omni_slam_odometry_eval_node PUBLIC ${OpenMP_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})

install(TARGETS omni_slam_tracking_eval_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS omni_slam_matching_eval_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS omni_slam_reconstruction_eval_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS omni_slam_odometry_eval_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
