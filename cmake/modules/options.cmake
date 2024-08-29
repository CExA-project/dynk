# testing
option(DYNK_ENABLE_TESTS "Build tests for the library")

# wrapper approach
option(DYNK_ENABLE_CXX20_FEATURES "Allow to use C++20 features" ON)

if(DYNK_ENABLE_CXX20_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

