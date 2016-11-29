cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(PREPROCESSING)
find_package(PCL 1.7 REQUIRED)
# find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(OpenCV REQUIRED )

include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(depth2pcl src/depth2pcl.cpp)

# add_executable(depth2pcl depth2pcl.cpp)
# target_link_libraries(depth2pcl ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so)

add_executable(people_detection src/people_detection.cpp)
target_link_libraries(people_detection depth2pcl ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so)
