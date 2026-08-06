cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(tango LANGUAGES CXX)
  option(CHRONOS_NATIVE_OPTIMIZATION "Compile for the build machine's CPU" OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX QUIET)

add_library(tango_headers INTERFACE)
target_include_directories(tango_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tango_headers INTERFACE Threads::Threads)

target_compile_definitions(tango_headers INTERFACE HAVE_CXX0X)
if(CHRONOS_NATIVE_OPTIMIZATION AND NOT MSVC)
  target_compile_options(tango_headers INTERFACE -march=native)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(tango_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(tango_index src/index.cpp)
target_link_libraries(tango_index PRIVATE tango_headers)

add_executable(tango_query src/query.cpp)
target_link_libraries(tango_query PRIVATE tango_headers)

add_executable(tango_dynamic src/dynamic.cpp)
target_link_libraries(tango_dynamic PRIVATE tango_headers)

add_executable(tango_decay_profile tools/decay_fastpath.cpp)
target_link_libraries(tango_decay_profile PRIVATE tango_headers)
