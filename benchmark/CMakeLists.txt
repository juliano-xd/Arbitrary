cmake_minimum_required(VERSION 3.14)
project(ArbitraryBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mbmi2 -madx")

include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(bench_arithmetic bench_arithmetic.cpp)
target_link_libraries(bench_arithmetic benchmark::benchmark)
target_include_directories(bench_arithmetic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(test_mul test_mul.cpp)
target_link_libraries(test_mul benchmark::benchmark)
target_include_directories(test_mul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
