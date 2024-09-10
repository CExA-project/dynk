add_library(
    dynk
    INTERFACE
)

target_sources(
    dynk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dynk/dual_view.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dynk/layer.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dynk/wrapper.hpp>
        $<INSTALL_INTERFACE:dynk/dual_view.hpp>
        $<INSTALL_INTERFACE:dynk/layer.hpp>
        $<INSTALL_INTERFACE:dynk/wrapper.hpp>
)

target_include_directories(
    dynk
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:.>
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
