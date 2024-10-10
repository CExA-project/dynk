include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
    TARGETS
        dynk
    EXPORT
        DynkTargets
    ARCHIVE DESTINATION
        "${CMAKE_INSTALL_LIBDIR}"
)

install(
    EXPORT
        DynkTargets
    NAMESPACE Dynk::
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/Dynk"
)

configure_package_config_file( 
    "${PROJECT_SOURCE_DIR}/cmake/modules/Config.cmake.in" 
    "${CMAKE_CURRENT_BINARY_DIR}/DynkConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Dynk"
    PATH_VARS
        CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/DynkConfigVersion.cmake"
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/DynkConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/DynkConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/Dynk"
)
