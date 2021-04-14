cmake_minimum_required(VERSION 3.17)
project(segment)

set(CMAKE_CXX_STANDARD 14)
find_package (OpenCV 4 REQUIRED)

include_directories( include
                    ${OpenCV_DIRS}
                    ${EIGEN3_INCLUDE_DIRS})

add_executable(segment main.cpp )
target_link_libraries(segment ${OpenCV_LIBS})