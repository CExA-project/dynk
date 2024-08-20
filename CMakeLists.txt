cmake_minimum_required(VERSION 3.22)

project(
    DynamicKokkos
    LANGUAGES CXX
    VERSION 0.1.0
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

include(options)
include(dependencies)

add_subdirectory(include)

include(install)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
