cmake_minimum_required(VERSION 3.12)
project(Labeling-pixels-in-a-2-dimensional-array)

set(CMAKE_CXX_STANDARD 14)
file(GLOB source_files
        "*.h"
        "*.cpp"
        "src/*.cpp"
        "include/*.h"
        )


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(serial ${source_files})
target_link_libraries( serial ${OpenCV_LIBS})