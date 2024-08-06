cmake_minimum_required(VERSION 3.10)
project(NeighborhoodCorrelation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(myexe main.cpp)

target_link_libraries(myexe PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(myexe PRIVATE -O3)
endif()