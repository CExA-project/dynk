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

add_executable(
    test-wrapper
    main.cpp
    test_wrapper.cpp
)

target_link_libraries(
    test-wrapper
    Dynk::dynk
    GTest::gtest
)
