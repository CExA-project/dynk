add_executable(
    example-layer
    example_layer.cpp
)

target_link_libraries(
    example-layer
    Dynk::dynk
)

add_executable(
    example-if-function-twice
    example_if_function_twice.cpp
)

target_link_libraries(
    example-if-function-twice
    Dynk::dynk
)

add_executable(
    example-wrapper-function-twice
    example_wrapper_function_twice.cpp
)

target_link_libraries(
    example-wrapper-function-twice
    Dynk::dynk
)

if(DYNK_ENABLE_CXX20_FEATURES)
    add_executable(
        example-wrapper-functor
        example_wrapper_functor.cpp
    )

    target_link_libraries(
        example-wrapper-functor
        Dynk::dynk
    )

    add_executable(
        example-wrapper-function
        example_wrapper_function.cpp
    )

    target_link_libraries(
        example-wrapper-function
        Dynk::dynk
    )

    if(DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA)
        add_executable(
            example-wrapper-lambda
            example_wrapper_lambda.cpp
        )

        target_link_libraries(
            example-wrapper-lambda
            Dynk::dynk
        )
    endif()
endif()
