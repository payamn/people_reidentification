# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/faraz/workspace/people_reidentification/depth2pcd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/faraz/workspace/people_reidentification/depth2pcd/build

# Include any dependencies generated for this target.
include CMakeFiles/depth2pcd.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/depth2pcd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/depth2pcd.dir/flags.make

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o: CMakeFiles/depth2pcd.dir/flags.make
CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o: ../depth2pcd.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/faraz/workspace/people_reidentification/depth2pcd/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o -c /home/faraz/workspace/people_reidentification/depth2pcd/depth2pcd.cpp

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/depth2pcd.dir/depth2pcd.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/faraz/workspace/people_reidentification/depth2pcd/depth2pcd.cpp > CMakeFiles/depth2pcd.dir/depth2pcd.cpp.i

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/depth2pcd.dir/depth2pcd.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/faraz/workspace/people_reidentification/depth2pcd/depth2pcd.cpp -o CMakeFiles/depth2pcd.dir/depth2pcd.cpp.s

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.requires:
.PHONY : CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.requires

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.provides: CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.requires
	$(MAKE) -f CMakeFiles/depth2pcd.dir/build.make CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.provides.build
.PHONY : CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.provides

CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.provides.build: CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o

# Object files for target depth2pcd
depth2pcd_OBJECTS = \
"CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o"

# External object files for target depth2pcd
depth2pcd_EXTERNAL_OBJECTS =

depth2pcd: CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o
depth2pcd: CMakeFiles/depth2pcd.dir/build.make
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_system.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_thread.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libpthread.so
depth2pcd: /usr/lib/libpcl_common.so
depth2pcd: /usr/lib/libOpenNI.so
depth2pcd: /usr/lib/libvtkCommon.so.5.8.0
depth2pcd: /usr/lib/libvtkRendering.so.5.8.0
depth2pcd: /usr/lib/libvtkHybrid.so.5.8.0
depth2pcd: /usr/lib/libvtkCharts.so.5.8.0
depth2pcd: /usr/lib/libpcl_io.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_system.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_thread.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libpthread.so
depth2pcd: /usr/lib/libpcl_common.so
depth2pcd: /usr/lib/libpcl_octree.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_system.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_thread.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libpthread.so
depth2pcd: /usr/lib/libpcl_common.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_system.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_thread.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libpthread.so
depth2pcd: /usr/lib/libpcl_common.so
depth2pcd: /usr/lib/libpcl_octree.so
depth2pcd: /usr/lib/libOpenNI.so
depth2pcd: /usr/lib/libvtkCommon.so.5.8.0
depth2pcd: /usr/lib/libvtkRendering.so.5.8.0
depth2pcd: /usr/lib/libvtkHybrid.so.5.8.0
depth2pcd: /usr/lib/libvtkCharts.so.5.8.0
depth2pcd: /usr/lib/libpcl_io.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
depth2pcd: /usr/lib/libpcl_kdtree.so
depth2pcd: /usr/lib/libpcl_search.so
depth2pcd: /usr/lib/libpcl_visualization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_system.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_thread.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libpthread.so
depth2pcd: /usr/lib/libOpenNI.so
depth2pcd: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
depth2pcd: /usr/lib/libvtkCommon.so.5.8.0
depth2pcd: /usr/lib/libvtkRendering.so.5.8.0
depth2pcd: /usr/lib/libvtkHybrid.so.5.8.0
depth2pcd: /usr/lib/libvtkCharts.so.5.8.0
depth2pcd: /usr/lib/libpcl_common.so
depth2pcd: /usr/lib/libpcl_io.so
depth2pcd: /usr/lib/libpcl_octree.so
depth2pcd: /usr/lib/libpcl_kdtree.so
depth2pcd: /usr/lib/libpcl_search.so
depth2pcd: /usr/lib/libpcl_visualization.so
depth2pcd: /usr/lib/libvtkViews.so.5.8.0
depth2pcd: /usr/lib/libvtkInfovis.so.5.8.0
depth2pcd: /usr/lib/libvtkWidgets.so.5.8.0
depth2pcd: /usr/lib/libvtkHybrid.so.5.8.0
depth2pcd: /usr/lib/libvtkParallel.so.5.8.0
depth2pcd: /usr/lib/libvtkVolumeRendering.so.5.8.0
depth2pcd: /usr/lib/libvtkRendering.so.5.8.0
depth2pcd: /usr/lib/libvtkGraphics.so.5.8.0
depth2pcd: /usr/lib/libvtkImaging.so.5.8.0
depth2pcd: /usr/lib/libvtkIO.so.5.8.0
depth2pcd: /usr/lib/libvtkFiltering.so.5.8.0
depth2pcd: /usr/lib/libvtkCommon.so.5.8.0
depth2pcd: /usr/lib/libvtksys.so.5.8.0
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
depth2pcd: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
depth2pcd: CMakeFiles/depth2pcd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable depth2pcd"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/depth2pcd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/depth2pcd.dir/build: depth2pcd
.PHONY : CMakeFiles/depth2pcd.dir/build

CMakeFiles/depth2pcd.dir/requires: CMakeFiles/depth2pcd.dir/depth2pcd.cpp.o.requires
.PHONY : CMakeFiles/depth2pcd.dir/requires

CMakeFiles/depth2pcd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/depth2pcd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/depth2pcd.dir/clean

CMakeFiles/depth2pcd.dir/depend:
	cd /home/faraz/workspace/people_reidentification/depth2pcd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/faraz/workspace/people_reidentification/depth2pcd /home/faraz/workspace/people_reidentification/depth2pcd /home/faraz/workspace/people_reidentification/depth2pcd/build /home/faraz/workspace/people_reidentification/depth2pcd/build /home/faraz/workspace/people_reidentification/depth2pcd/build/CMakeFiles/depth2pcd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/depth2pcd.dir/depend

