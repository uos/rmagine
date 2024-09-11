
message(STATUS "Building Core. Library: rmagine")

set(RMAGINE_CORE_SRCS
    # Types
    src/types/ouster_sensors.cpp
)

add_library(rmagine-ouster SHARED
    ${RMAGINE_CORE_SRCS}
)

target_include_directories(rmagine-ouster
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
    ${ASSIMP_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(rmagine-ouster
    rmagine-core
    ${ASSIMP_LIBRARIES}
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    JsonCpp::JsonCpp
)

set_target_properties(rmagine-ouster
  PROPERTIES
    EXPORT_NAME ouster
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
    COMPONENT ouster
    CXX_STANDARD 17
)

# TODO: do this:
add_library(rmagine ALIAS rmagine-ouster)
add_library(rmagine::ouster ALIAS rmagine-ouster)


list(APPEND RMAGINE_LIBRARIES rmagine-ouster)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)


# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

###########
## CORE
############
install(TARGETS rmagine-ouster 
  EXPORT rmagine-ouster-targets
  COMPONENT ouster)

install(EXPORT rmagine-ouster-targets
    FILE rmagine-ouster-targets.cmake
    COMPONENT ouster
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-ouster-config-version.cmake
  VERSION ${rmagine_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-ouster-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-ouster-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-ouster-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-ouster-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
  COMPONENT ouster
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT ouster
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libeigen3-dev, libassimp-dev" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL ouster)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)

