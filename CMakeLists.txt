cmake_minimum_required(VERSION 3.21)

project(ShibuyaStream)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Set build type to Release by default" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

enable_language(HIP)

add_compile_options(-mavx)

find_library(NUMA_LIBRARY NAMES numa)

file(GLOB SOURCES "src/*.cpp")

set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(shibuya ${SOURCES})

target_link_libraries(shibuya PRIVATE
    ${NUMA_LIBRARY} -pthread)
