add_library(
    dynk
    INTERFACE
)

target_include_directories(
    dynk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    dynk
    INTERFACE
        Kokkos::kokkos
)

install(
    DIRECTORY
        "${CMAKE_CURRENT_LIST_DIR}/dynk"
    DESTINATION
        include
)

add_library(
    Dynk::dynk
    ALIAS
        dynk
)
