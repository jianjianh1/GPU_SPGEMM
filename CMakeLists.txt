cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gpu_spgemm VERSION 1.0 LANGUAGES CXX CUDA)

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
evaluate_gpu_archs(GPU_ARCHS)

find_package(CUDAToolkit 11.8 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

set(CPM_SOURCE_CACHE ${CMAKE_CURRENT_SOURCE_DIR}/cache)
include(cmake/CPM.cmake)

CPMAddPackage(
  NAME argparse
  GITHUB_REPOSITORY p-ranav/argparse
  GIT_TAG origin/master
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcudart --extended-lambda")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LINFO "Add lineInfo for NSight Compute profiling" OFF)
if(LINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

option(PTXAS_VERBOSE "Enable verbose PTX assembly output" OFF)
if(PTXAS_VERBOSE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif()

add_executable(spgemm_raw_v2 src/spgemm_raw_v2.cu)
set_target_properties(spgemm_raw_v2 PROPERTIES
  CUDA_ARCHITECTURES ${GPU_ARCHS}
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(spgemm_raw_v2 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gpu_error/include
)
target_link_libraries(spgemm_raw_v2 PRIVATE argparse cudadevrt)
target_compile_options(spgemm_raw_v2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
