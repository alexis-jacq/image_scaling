cmake_minimum_required(VERSION 2.8)

project(IMAGE_SCALING)
find_package(OpenCV REQUIRED )
add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(DCCI src/dcci.cpp)
add_executable(artisticAM src/artisticAM.cpp)
add_executable(AM src/AM.cpp)
target_link_libraries(DCCI ${OpenCV_LIBS})
target_link_libraries(artisticAM ${OpenCV_LIBS})
target_link_libraries(AM ${OpenCV_LIBS})

install(TARGETS DCCI RUNTIME DESTINATION bin)
install(TARGETS artisticAM RUNTIME DESTINATION bin)
install(TARGETS AM RUNTIME DESTINATION bin)
