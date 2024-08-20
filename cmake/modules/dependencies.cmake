include(FetchContent)

find_package(Kokkos 4.3.1 QUIET)
if(NOT Kokkos_FOUND)
    message(STATUS "Treating Kokkos as an internal dependency")
    FetchContent_Declare(
        kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz
    )
    FetchContent_MakeAvailable(kokkos)
endif()

if(ENABLE_TESTS)
    find_package(googletest 1.15.2 QUIET)
    if(NOT googletest_FOUND)
        message(STATUS "Treating Gtest as an internal dependency")
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
