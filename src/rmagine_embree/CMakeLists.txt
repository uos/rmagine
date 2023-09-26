##########################
## Package: rmagine     ##
## Component: embree    ##
##########################

message(STATUS "Building Embree (${embree_VERSION}) backend. Library: rmagine-embree")

set(RMAGINE_EMBREE_SRCS
    # Maps
    src/map/embree/EmbreeDevice.cpp
    src/map/embree/EmbreeGeometry.cpp
    src/map/embree/EmbreeMesh.cpp
    src/map/embree/EmbreeScene.cpp
    src/map/embree/EmbreeInstance.cpp
    src/map/embree/EmbreePoints.cpp
    src/map/embree/embree_shapes.cpp
    src/map/EmbreeMap.cpp

    # Simulators
    src/simulation/SphereSimulatorEmbree.cpp
    src/simulation/PinholeSimulatorEmbree.cpp
    src/simulation/O1DnSimulatorEmbree.cpp
    src/simulation/OnDnSimulatorEmbree.cpp
)

## SHARED ##
add_library(rmagine-embree SHARED
    ${RMAGINE_EMBREE_SRCS}
)

target_include_directories(rmagine-embree
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rmagine-embree
    rmagine-core
    ${embree_LIBRARY}
)

add_dependencies(rmagine-embree
    rmagine-core
)

target_compile_features(rmagine-embree PRIVATE cxx_std_17)

set_target_properties(rmagine-embree
    PROPERTIES
    EXPORT_NAME embree
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::embree ALIAS rmagine-embree)

list(APPEND RMAGINE_LIBRARIES rmagine-embree)

## STATIC
add_library(rmagine-embree-static STATIC
    ${RMAGINE_EMBREE_SRCS}
)

target_include_directories(rmagine-embree-static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


target_link_libraries(rmagine-embree-static
    rmagine-core-static
    ${embree_LIBRARY}
)

add_dependencies(rmagine-embree-static
    rmagine-core-static
)

target_compile_features(rmagine-embree-static PRIVATE cxx_std_17)

set_target_properties(rmagine-embree-static
    PROPERTIES
    EXPORT_NAME embree-static
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::embree-static ALIAS rmagine-embree-static)

list(APPEND RMAGINE_STATIC_LIBRARIES rmagine-embree-static)


###########
## INSTALL
############

include(CMakePackageConfigHelpers)

install(TARGETS rmagine-embree EXPORT rmagine-embree-targets)

install(EXPORT rmagine-embree-targets
    FILE rmagine-embree-targets.cmake
    COMPONENT embree
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-embree-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
  COMPONENT embree
)

install(
  DIRECTORY include/rmagine
  COMPONENT embree
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS rmagine-embree-static EXPORT rmagine-embree-static-targets)

install(EXPORT rmagine-embree-static-targets
    FILE rmagine-embree-static-targets.cmake
    COMPONENT embree-static
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-static-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-embree-static-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-static-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-static-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-static-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
  COMPONENT embree-static
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT embree-static
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)