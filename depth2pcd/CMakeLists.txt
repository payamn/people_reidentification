cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DEPTH2PCD)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_executable(depth2pcd depth2pcd.cpp)
# add_executable(people_extractor people_extractor.cpp)

target_link_libraries(depth2pcd ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so)
# target_link_libraries(people_extractor ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so)
