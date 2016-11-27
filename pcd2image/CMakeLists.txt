cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(pcd2image)
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd2image MACOSX_BUNDLE pcd2image.cpp)
target_link_libraries (pcd2image ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
