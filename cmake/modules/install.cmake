include(GNUInstallDirs)

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
        "${CMAKE_INSTALL_LIBDIR}/Dynk"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DynkConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/DynkConfigVersion.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/Dynk"
)
