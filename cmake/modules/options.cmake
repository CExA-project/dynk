# testing
option(DYNK_ENABLE_TESTS "Build tests for the library")

# wrapper approach
option(DYNK_ENABLE_CXX20_FEATURES "Allow to use C++20 features" ON)

if(DYNK_ENABLE_CXX20_FEATURES)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

option(DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA "Allow to have an extended lambda defined inside a generic lambda, requires C++20")

if(DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA)
    if(NOT DYNK_ENABLE_CXX20_FEATURES)
        message(FATAL_ERROR "Extended lambda in generic lambda feature requires C++20 standard enabled")
    endif()
endif()

# allow gtest to discover tests
option(DYNK_ENABLE_GTEST_DISCOVER_TESTS "Enable Gtest to discover tests by attempting to run them" ON)
