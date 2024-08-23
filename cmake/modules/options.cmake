# testing
option(DYNK_ENABLE_TESTS "Build tests for the library")

# wrapper approach
option(DYNK_ENABLE_WRAPPER "Allow to use the wrapper approach that requires C++20" ON)

if(DYNK_ENABLE_WRAPPER)
    set(CMAKE_CXX_STANDARD 20)
endif()

