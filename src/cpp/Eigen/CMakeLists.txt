cmake_minimum_required(VERSION 3.10)
project(NeighborhoodCorrelation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

add_executable(nc_program main.cpp)
target_include_directories(nc_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nc_program PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(nc_program PRIVATE OpenMP::OpenMP_CXX)
endif()
