# set Doxyfile paths
set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# set custom Doxygen parameters
set(DOXYGEN_INPUT "${PROJECT_SOURCE_DIR}/include/dynk")
if(DYNK_ENABLE_DOCUMENTATION_DEVMODE)
    set(DOXYGEN_DEVMODE "YES")
else()
    set(DOXYGEN_DEVMODE "NO")
endif()
set(DOXYGEN_AWESOME_CSS_PATH "${awesome_doxygen_css_SOURCE_DIR}")

configure_file(
    ${DOXYGEN_IN}
    ${DOXYGEN_OUT}
    @ONLY
)

add_custom_target(
    docs
    ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)
