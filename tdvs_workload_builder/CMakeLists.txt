cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(chronos_tdvs_workload_builder LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX QUIET)

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE CHRONOS_GIT_RESULT
  OUTPUT_VARIABLE CHRONOS_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT CHRONOS_GIT_RESULT EQUAL 0)
  set(CHRONOS_GIT_REVISION "unknown")
endif()
execute_process(
  COMMAND git status --porcelain --untracked-files=normal
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE CHRONOS_GIT_STATUS_RESULT
  OUTPUT_VARIABLE CHRONOS_GIT_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT CHRONOS_GIT_STATUS_RESULT EQUAL 0)
  set(CHRONOS_GIT_DIRTY -1)
elseif(CHRONOS_GIT_STATUS STREQUAL "")
  set(CHRONOS_GIT_DIRTY 0)
else()
  set(CHRONOS_GIT_DIRTY 1)
endif()

add_library(chronos_workload_core
  src/alignment.cpp
  src/io.cpp
  src/sampling.cpp
  src/time_metadata.cpp
  src/tdvs.cpp
  src/validation.cpp
)
target_include_directories(chronos_workload_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(chronos_workload_core PUBLIC Threads::Threads)
target_compile_definitions(chronos_workload_core PRIVATE
  CHRONOS_GIT_REVISION="${CHRONOS_GIT_REVISION}"
  CHRONOS_GIT_DIRTY=${CHRONOS_GIT_DIRTY}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(chronos_workload_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(chronos_workload_core PUBLIC CHRONOS_HAVE_OPENMP=1)
endif()

function(chronos_workload_tool exe)
  add_executable(${exe} tools/${exe}.cpp)
  target_link_libraries(${exe} PRIVATE chronos_workload_core)
endfunction()

chronos_workload_tool(prepare_vector_subset)
chronos_workload_tool(generate_time_metadata)
chronos_workload_tool(generate_tdvs_groundtruth)
chronos_workload_tool(export_mips_transforms)
chronos_workload_tool(verify_dataset_alignment)
