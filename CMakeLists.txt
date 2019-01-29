cmake_minimum_required(VERSION 3.12)
project(Labeling-pixels-in-a-2-dimensional-array)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(output main.cpp)
target_link_libraries( output ${OpenCV_LIBS})