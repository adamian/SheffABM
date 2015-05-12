cmake_minimum_required(VERSION 2.8.9)
project( DisplayImage )

SET(projectName webcamToYarp)

find_package( OpenCV REQUIRED )
find_package( YARP REQUIRED )

file(GLOB folder_source *.cpp)
file(GLOB folder_header *.h)

set(CMAKE_CXX_FLAGS " ")
source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})

INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${YARP_INCLUDE_DIRS})

add_executable( ${projectName} ${folder_source} ${folder_header})
target_link_libraries( ${projectName} ${OpenCV_LIBS} ${YARP_LIBRARIES} ${CMAKE_CX_FLAGS} /usr/lib/x86_64-linux-gnu/libv4l2.a)
install(TARGETS ${projectName} DESTINATION bin)
