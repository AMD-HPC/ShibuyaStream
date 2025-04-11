# ==============================================================================
# \file
# \brief      CMake configuration for building the ShibuyaStream project
# \date       2020-2024
# \author     Jakub Kurzak
# \copyright  Advanced Micro Devices, Inc.
#
cmake_minimum_required(VERSION 3.21)

project(ShibuyaStream)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Set build type to Release by default" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options to choose between HIP and CUDA
option(USE_HIP "Use HIP for AMD GPUs" ON)
option(USE_CUDA "Use CUDA for NVIDIA GPUs" OFF)

# Option to enable AVX support
option(ENABLE_AVX_SUPPORT "Enable AVX instructions" ON)

if(USE_HIP AND USE_CUDA)
  message(FATAL_ERROR "Cannot use both HIP and CUDA at the same time.")

# ------------------------------------------------------------------------------
# Setup for building with HIP (AMD GPUs)
#
elseif(USE_HIP)
  enable_language(HIP)
  find_library(NUMA_LIBRARY NAMES numa)

  file(GLOB SOURCES "src/*.cpp")
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

  add_executable(shibuya ${SOURCES})

  target_link_libraries(shibuya PRIVATE ${NUMA_LIBRARY} -pthread)
  target_compile_definitions(shibuya PRIVATE USE_HIP)

  if(ENABLE_AVX_SUPPORT)
    target_compile_definitions(shibuya PRIVATE ENABLE_AVX_SUPPORT)
    target_compile_options(shibuya PRIVATE -mavx)
  endif()

# ------------------------------------------------------------------------------
# Setup for building with CUDA (NVIDIA GPUs)
#
elseif(USE_CUDA)
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_CUDA_COMPILER clang++)

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  file(GLOB SOURCES "src/*.cpp")
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

  add_executable(shibuya ${SOURCES})

  target_link_libraries(shibuya PRIVATE ${NUMA_LIBRARY} -pthread)
  target_compile_definitions(shibuya PRIVATE USE_CUDA)

  if(ENABLE_AVX_SUPPORT)
    target_compile_definitions(shibuya PRIVATE ENABLE_AVX_SUPPORT)
    target_compile_options(shibuya PRIVATE -mavx)
  endif()

else()
  message(FATAL_ERROR "Either USE_HIP or USE_CUDA must be set.")

endif()
