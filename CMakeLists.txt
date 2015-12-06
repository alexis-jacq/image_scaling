cmake_minimum_required(VERSION 2.8)

#Déclaration du projet
project(IMAGE_SCALING)

find_package(OpenCV REQUIRED )

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
#Déclaration de l'exécutable
add_executable(scaling src/scale.cpp)
target_link_libraries(scaling ${OpenCV_LIBS})


install(TARGETS scaling RUNTIME DESTINATION bin)
