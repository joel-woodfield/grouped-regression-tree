cmake_minimum_required(VERSION 3.15)
project(grouped_regression_tree)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(grouped_regression_tree
    src/bindings.cpp
    src/GroupedRegressionTree.cpp
)

target_include_directories(grouped_regression_tree PRIVATE include)

target_link_libraries(grouped_regression_tree PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(grouped_regression_tree PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2>

    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -O3
        -march=native
        -ffast-math
        -funroll-loops
    >
)

install(TARGETS grouped_regression_tree DESTINATION .)

