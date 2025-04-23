cmake_minimum_required(VERSION 3.22)

project(
    Dynk
    LANGUAGES CXX
    VERSION 0.4.0
)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

include(options)
include(dependencies)

add_subdirectory(include)

include(install)

if(DYNK_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(DYNK_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(DYNK_ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif()
