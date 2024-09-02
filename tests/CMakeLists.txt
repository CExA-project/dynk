add_executable(
    test-layer
    main.cpp
    test_layer.cpp
)

target_link_libraries(
    test-layer
    Dynk::dynk
    GTest::gtest
)

gtest_discover_tests(test-layer)

add_executable(
    test-wrapper
    main.cpp
    test_wrapper.cpp
)

target_compile_definitions(
    test-wrapper
    PRIVATE
        $<IF:$<BOOL:${DYNK_ENABLE_CXX20_FEATURES}>,ENABLE_CXX20_FEATURES,>
        $<IF:$<BOOL:${DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA}>,ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA,>
)

target_link_libraries(
    test-wrapper
    Dynk::dynk
    GTest::gtest
)

gtest_discover_tests(test-wrapper)
