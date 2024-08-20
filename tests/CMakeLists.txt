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
