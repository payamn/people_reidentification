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
CMAKE_SOURCE_DIR = /home/faraz/workspace/people_reidentification/body_extractor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/faraz/workspace/people_reidentification/body_extractor/build

# Include any dependencies generated for this target.
include CMakeFiles/extractor.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/extractor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/extractor.dir/flags.make

CMakeFiles/extractor.dir/extractor.cpp.o: CMakeFiles/extractor.dir/flags.make
CMakeFiles/extractor.dir/extractor.cpp.o: ../extractor.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/faraz/workspace/people_reidentification/body_extractor/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/extractor.dir/extractor.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/extractor.dir/extractor.cpp.o -c /home/faraz/workspace/people_reidentification/body_extractor/extractor.cpp

CMakeFiles/extractor.dir/extractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/extractor.dir/extractor.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/faraz/workspace/people_reidentification/body_extractor/extractor.cpp > CMakeFiles/extractor.dir/extractor.cpp.i

CMakeFiles/extractor.dir/extractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/extractor.dir/extractor.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/faraz/workspace/people_reidentification/body_extractor/extractor.cpp -o CMakeFiles/extractor.dir/extractor.cpp.s

CMakeFiles/extractor.dir/extractor.cpp.o.requires:
.PHONY : CMakeFiles/extractor.dir/extractor.cpp.o.requires

CMakeFiles/extractor.dir/extractor.cpp.o.provides: CMakeFiles/extractor.dir/extractor.cpp.o.requires
	$(MAKE) -f CMakeFiles/extractor.dir/build.make CMakeFiles/extractor.dir/extractor.cpp.o.provides.build
.PHONY : CMakeFiles/extractor.dir/extractor.cpp.o.provides

CMakeFiles/extractor.dir/extractor.cpp.o.provides.build: CMakeFiles/extractor.dir/extractor.cpp.o

# Object files for target extractor
extractor_OBJECTS = \
"CMakeFiles/extractor.dir/extractor.cpp.o"

# External object files for target extractor
extractor_EXTERNAL_OBJECTS =

extractor: CMakeFiles/extractor.dir/extractor.cpp.o
extractor: CMakeFiles/extractor.dir/build.make
extractor: /usr/lib/x86_64-linux-gnu/libboost_system.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_thread.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
extractor: /usr/lib/x86_64-linux-gnu/libpthread.so
extractor: /usr/lib/libpcl_common.so
extractor: /usr/lib/libpcl_octree.so
extractor: /usr/lib/libOpenNI.so
extractor: /usr/lib/libvtkCommon.so.5.8.0
extractor: /usr/lib/libvtkRendering.so.5.8.0
extractor: /usr/lib/libvtkHybrid.so.5.8.0
extractor: /usr/lib/libvtkCharts.so.5.8.0
extractor: /usr/lib/libpcl_io.so
extractor: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
extractor: /usr/lib/libpcl_kdtree.so
extractor: /usr/lib/libpcl_search.so
extractor: /usr/lib/libpcl_sample_consensus.so
extractor: /usr/lib/libpcl_filters.so
extractor: /usr/lib/libpcl_features.so
extractor: /usr/lib/libpcl_keypoints.so
extractor: /usr/lib/libpcl_segmentation.so
extractor: /usr/lib/libpcl_visualization.so
extractor: /usr/lib/libpcl_outofcore.so
extractor: /usr/lib/libpcl_registration.so
extractor: /usr/lib/libpcl_recognition.so
extractor: /usr/lib/x86_64-linux-gnu/libqhull.so
extractor: /usr/lib/libpcl_surface.so
extractor: /usr/lib/libpcl_people.so
extractor: /usr/lib/libpcl_tracking.so
extractor: /usr/lib/libpcl_apps.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_system.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_thread.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
extractor: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
extractor: /usr/lib/x86_64-linux-gnu/libpthread.so
extractor: /usr/lib/x86_64-linux-gnu/libqhull.so
extractor: /usr/lib/libOpenNI.so
extractor: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
extractor: /usr/lib/libvtkCommon.so.5.8.0
extractor: /usr/lib/libvtkRendering.so.5.8.0
extractor: /usr/lib/libvtkHybrid.so.5.8.0
extractor: /usr/lib/libvtkCharts.so.5.8.0
extractor: /usr/lib/libpcl_common.so
extractor: /usr/lib/libpcl_octree.so
extractor: /usr/lib/libpcl_io.so
extractor: /usr/lib/libpcl_kdtree.so
extractor: /usr/lib/libpcl_search.so
extractor: /usr/lib/libpcl_sample_consensus.so
extractor: /usr/lib/libpcl_filters.so
extractor: /usr/lib/libpcl_features.so
extractor: /usr/lib/libpcl_keypoints.so
extractor: /usr/lib/libpcl_segmentation.so
extractor: /usr/lib/libpcl_visualization.so
extractor: /usr/lib/libpcl_outofcore.so
extractor: /usr/lib/libpcl_registration.so
extractor: /usr/lib/libpcl_recognition.so
extractor: /usr/lib/libpcl_surface.so
extractor: /usr/lib/libpcl_people.so
extractor: /usr/lib/libpcl_tracking.so
extractor: /usr/lib/libpcl_apps.so
extractor: /usr/lib/libvtkViews.so.5.8.0
extractor: /usr/lib/libvtkInfovis.so.5.8.0
extractor: /usr/lib/libvtkWidgets.so.5.8.0
extractor: /usr/lib/libvtkHybrid.so.5.8.0
extractor: /usr/lib/libvtkParallel.so.5.8.0
extractor: /usr/lib/libvtkVolumeRendering.so.5.8.0
extractor: /usr/lib/libvtkRendering.so.5.8.0
extractor: /usr/lib/libvtkGraphics.so.5.8.0
extractor: /usr/lib/libvtkImaging.so.5.8.0
extractor: /usr/lib/libvtkIO.so.5.8.0
extractor: /usr/lib/libvtkFiltering.so.5.8.0
extractor: /usr/lib/libvtkCommon.so.5.8.0
extractor: /usr/lib/libvtksys.so.5.8.0
extractor: CMakeFiles/extractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable extractor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/extractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/extractor.dir/build: extractor
.PHONY : CMakeFiles/extractor.dir/build

CMakeFiles/extractor.dir/requires: CMakeFiles/extractor.dir/extractor.cpp.o.requires
.PHONY : CMakeFiles/extractor.dir/requires

CMakeFiles/extractor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/extractor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/extractor.dir/clean

CMakeFiles/extractor.dir/depend:
	cd /home/faraz/workspace/people_reidentification/body_extractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/faraz/workspace/people_reidentification/body_extractor /home/faraz/workspace/people_reidentification/body_extractor /home/faraz/workspace/people_reidentification/body_extractor/build /home/faraz/workspace/people_reidentification/body_extractor/build /home/faraz/workspace/people_reidentification/body_extractor/build/CMakeFiles/extractor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/extractor.dir/depend

