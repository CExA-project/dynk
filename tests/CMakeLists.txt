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
    test-dynamic
    main.cpp
    test_dynamic.cpp
)

target_link_libraries(
    test-dynamic
    Dynk::dynk
    GTest::gtest
)
