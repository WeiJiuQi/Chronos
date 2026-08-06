cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(chronos_baselines LANGUAGES CXX)
  option(CHRONOS_NATIVE_OPTIMIZATION "Compile for the build machine's CPU" OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP COMPONENTS CXX QUIET)
find_package(Threads REQUIRED)

# Hash each runner's registered provenance source set rather than only recording
# an upstream commit. This captures the portability, correctness, and runner
# patches in the source release. A sorted per-file SHA-256 manifest is also
# emitted for independent auditing.
set(TDVS_PROVENANCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

function(tdvs_collect_source_files out_var)
  set(collected)
  foreach(root IN LISTS ARGN)
    if(IS_DIRECTORY "${root}")
      file(GLOB_RECURSE root_sources CONFIGURE_DEPENDS LIST_DIRECTORIES false
        "${root}/*.c" "${root}/*.cc" "${root}/*.cpp" "${root}/*.cxx"
        "${root}/*.h" "${root}/*.hh" "${root}/*.hpp" "${root}/*.hxx")
      list(APPEND collected ${root_sources})
    elseif(EXISTS "${root}")
      list(APPEND collected "${root}")
    else()
      message(FATAL_ERROR "Provenance source does not exist: ${root}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES collected)
  list(SORT collected)
  set(${out_var} "${collected}" PARENT_SCOPE)
endfunction()

function(tdvs_set_source_provenance target)
  tdvs_collect_source_files(source_files ${ARGN})
  if(NOT source_files)
    message(FATAL_ERROR "No provenance sources registered for ${target}")
  endif()
  # Existing files are build-system inputs too: changing a header or runner
  # must rerun CMake before recompilation so the embedded hash cannot go stale.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${source_files})

  set(manifest_body "")
  foreach(source IN LISTS source_files)
    file(REAL_PATH "${source}" absolute_source)
    file(RELATIVE_PATH relative_source "${TDVS_PROVENANCE_ROOT}" "${absolute_source}")
    file(SHA256 "${absolute_source}" source_sha256)
    string(APPEND manifest_body "${source_sha256}  ${relative_source}\n")
  endforeach()
  string(SHA256 source_fingerprint "${manifest_body}")

  target_compile_definitions(${target} PRIVATE
    "TDVS_SOURCE_FINGERPRINT=\"${source_fingerprint}\"")
  set(manifest_dir "${CMAKE_CURRENT_BINARY_DIR}/source_provenance")
  file(MAKE_DIRECTORY "${manifest_dir}")
  file(WRITE "${manifest_dir}/${target}.manifest"
    "source_fingerprint=${source_fingerprint}\n${manifest_body}")
  message(STATUS "${target} source fingerprint: ${source_fingerprint}")
endfunction()

set(TDVS_SHARED_PROVENANCE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

# AppleClang's FindOpenMP frequently misses a Homebrew libomp installation.
if(APPLE AND NOT OpenMP_CXX_FOUND)
  find_path(TDVS_OMP_INCLUDE_DIR omp.h
    HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
  find_library(TDVS_OMP_LIBRARY omp
    HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
endif()

function(tdvs_configure_target target)
  target_compile_definitions(${target} PRIVATE HAVE_CXX0X)
  if(CHRONOS_NATIVE_OPTIMIZATION AND NOT MSVC)
    target_compile_options(${target} PRIVATE -march=native)
  endif()
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  elseif(APPLE AND TDVS_OMP_INCLUDE_DIR AND TDVS_OMP_LIBRARY)
    target_compile_options(${target} PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(${target} PRIVATE ${TDVS_OMP_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE ${TDVS_OMP_LIBRARY})
  else()
    message(WARNING "OpenMP unavailable: index construction will be serial")
  endif()

  target_link_libraries(${target} PRIVATE Threads::Threads)

endfunction()

add_executable(tdvs_mag runners/mag.cpp)
target_include_directories(tdvs_mag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mag/include)
tdvs_configure_target(tdvs_mag)
tdvs_set_source_provenance(tdvs_mag
  ${TDVS_SHARED_PROVENANCE_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/runners/mag.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mag/include")

add_executable(tdvs_psp
  runners/psp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index_random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index_mips.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/psp/src/util.cpp)
target_include_directories(tdvs_psp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/psp/include)
tdvs_configure_target(tdvs_psp)
tdvs_set_source_provenance(tdvs_psp
  ${TDVS_SHARED_PROVENANCE_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/runners/psp.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/psp/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index_random.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/psp/src/index_mips.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/psp/src/util.cpp")

add_executable(tdvs_ip_nsw_plus runners/ip_nsw_plus.cpp)
target_include_directories(tdvs_ip_nsw_plus PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw-plus)
tdvs_configure_target(tdvs_ip_nsw_plus)
tdvs_set_source_provenance(tdvs_ip_nsw_plus
  ${TDVS_SHARED_PROVENANCE_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/runners/ip_nsw_plus.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw-plus/L2space.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw-plus/hnswalg.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw-plus/hnswlib.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw-plus/visited_list_pool.h")

add_executable(tdvs_ip_nsw runners/ip_nsw.cpp)
target_include_directories(tdvs_ip_nsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw)
tdvs_configure_target(tdvs_ip_nsw)
tdvs_set_source_provenance(tdvs_ip_nsw
  ${TDVS_SHARED_PROVENANCE_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/runners/ip_nsw.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw/L2space.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw/hnswalg.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw/hnswlib.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ip-nsw/visited_list_pool.h")

# NAPG's neighbor-selection logic is a clean-room implementation of Tan et al.,
# KDD 2021 (DOI 10.1145/3447548.3467412). Its HNSW substrate remains isolated
# from ip-NSW so the NAPG-specific rule cannot affect another method.
add_executable(tdvs_napg runners/napg.cpp)
target_include_directories(tdvs_napg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/napg
  ${CMAKE_CURRENT_SOURCE_DIR}/napg/include)
tdvs_configure_target(tdvs_napg)
tdvs_set_source_provenance(tdvs_napg
  ${TDVS_SHARED_PROVENANCE_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/runners/napg.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/napg")
