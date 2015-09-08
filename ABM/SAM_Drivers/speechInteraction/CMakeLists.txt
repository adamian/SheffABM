cmake_minimum_required(VERSION 2.8)
set(PROJECTNAME speechInteraction)
project(${PROJECTNAME})

find_package(YARP REQUIRED)

file(GLOB folder_source src/*.cpp)
file(GLOB folder_header include/*.h)

source_group("Source Files" FILES ${folder_source})
source_group("Header Files" FILES ${folder_header})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${wrdac_INCLUDE_DIRS} ${YARP_INCLUDE_DIRS})
add_executable(${PROJECTNAME} ${folder_source} ${folder_header})
target_link_libraries(${PROJECTNAME} wrdac ${YARP_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER speech)
install(TARGETS ${PROJECTNAME} DESTINATION bin)
