add_executable(
    test-dynk
    main.cpp
    test_dynk.cpp
)

target_link_libraries(
    test-dynk
    Dynk::dynk
    GTest::gtest
)

gtest_discover_tests(test-dynk)

add_executable(
    test-wrapper
    main.cpp
    test_wrapper.cpp
)

target_compile_definitions(
    test-wrapper
    PRIVATE
        $<IF:$<BOOL:${DYNK_ENABLE_CXX20_FEATURES}>,ENABLE_CXX20_FEATURES,>
)

target_link_libraries(
    test-wrapper
    Dynk::dynk
    GTest::gtest
)

gtest_discover_tests(test-wrapper)
