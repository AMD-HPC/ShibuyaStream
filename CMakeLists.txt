cmake_minimum_required(VERSION 3.21)

project(ShibuyaStream)

enable_language(HIP)

add_compile_options(-mavx)

find_library(NUMA_LIBRARY NAMES numa)

file(GLOB SOURCES "src/*.cpp")

set_source_files_properties(
    ${SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(shibuya ${SOURCES})

target_link_libraries(shibuya PRIVATE
    ${NUMA_LIBRARY} -pthread)
