include(FetchContent)

find_package(Kokkos 4.6.0 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Treating Kokkos as an internal dependency")
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/releases/download/4.6.00/kokkos-4.6.00.tar.gz
        URL_HASH SHA256=be72cf7fc6ef6b99c614f29b945960013a2aaa23859bfe1a560d8d9aa526ec9c
    )
    FetchContent_MakeAvailable(kokkos)
endif()

if(DYNK_ENABLE_TESTS)
    find_package(googletest 1.16.0 QUIET)
    if(NOT googletest_FOUND)
        message(STATUS "Treating Gtest as an internal dependency")
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
            URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
        )
        FetchContent_MakeAvailable(googletest)
        include(GoogleTest)
    endif()
endif()

if(DYNK_ENABLE_DOCUMENTATION)
    find_package(Doxygen 1.9.1 REQUIRED QUIET)

    FetchContent_Declare(
        awesome_doxygen_css
        URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.3.4.tar.gz
        URL_HASH SHA256=54d515327b559cd18c6f4d3e8a46e0facc635be1967fb69526f977285922960e
    )
    FetchContent_MakeAvailable(awesome_doxygen_css)
endif()
