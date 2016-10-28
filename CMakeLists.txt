cmake_minimum_required(VERSION 2.8)

#Déclaration du projet
project(IMAGE_SCALING)

find_package(OpenCV REQUIRED )

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
#Déclaration de l'exécutable
add_executable(scaling src/scale.cpp)
add_executable(mosaic src/mosaic.cpp)
add_executable(entropy src/entropy.cpp)
add_executable(friends src/friends.cpp)
target_link_libraries(scaling ${OpenCV_LIBS})
target_link_libraries(entropy ${OpenCV_LIBS})
target_link_libraries(mosaic ${OpenCV_LIBS})
target_link_libraries(friends ${OpenCV_LIBS})

install(TARGETS scaling RUNTIME DESTINATION bin)
