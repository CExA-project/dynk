add_executable(
    example-1-layer
    example_1_layer.cpp
)

target_link_libraries(
    example-1-layer
    Dynk::dynk
)

add_executable(
    example-2-wrapper-if
    example_2_wrapper_if.cpp
)

target_link_libraries(
    example-2-wrapper-if
    Dynk::dynk
)

add_executable(
    example-3-wrapper-2-functions
    example_3_wrapper_2_functions.cpp
)

target_link_libraries(
    example-3-wrapper-2-functions
    Dynk::dynk
)

if(DYNK_ENABLE_CXX20_FEATURES)
    add_executable(
        example-4-wrapper-function
        example_4_wrapper_function.cpp
    )

    target_link_libraries(
        example-4-wrapper-function
        Dynk::dynk
    )

    add_executable(
        example-5-wrapper-functor
        example_5_wrapper_functor.cpp
    )

    target_link_libraries(
        example-5-wrapper-functor
        Dynk::dynk
    )

    if(DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA)
        add_executable(
            example-6-wrapper-lambda
            example_6_wrapper_lambda.cpp
        )

        target_link_libraries(
            example-6-wrapper-lambda
            Dynk::dynk
        )
    endif()
endif()
