cmake_minimum_required(VERSION 3.21)
project(chronos LANGUAGES CXX)

option(CHRONOS_BUILD_TANGO "Build the TANGO index" ON)
option(CHRONOS_BUILD_TDVS_WORKLOAD_BUILDER "Build the TDVS workload tools" ON)
option(CHRONOS_BUILD_BASELINES "Build the five MIPS baselines" ON)
option(CHRONOS_NATIVE_OPTIMIZATION "Compile for the build machine's CPU" OFF)

if(CHRONOS_BUILD_TANGO)
  add_subdirectory(tango)
endif()

if(CHRONOS_BUILD_TDVS_WORKLOAD_BUILDER)
  add_subdirectory(tdvs_workload_builder)
endif()

if(CHRONOS_BUILD_BASELINES)
  add_subdirectory(baselines)
endif()
