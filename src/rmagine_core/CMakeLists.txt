
message(STATUS "Building Core. Library: rmagine")

set(RMAGINE_CORE_SRCS
    # Maps
    src/map/AssimpIO.cpp
    # # Math
    src/math/math.cpp
    src/math/linalg.cpp
    src/math/SVD.cpp
    # Types
    src/types/Memory.cpp
    src/types/conversions.cpp
    src/types/sensors.cpp
    # Util
    src/util/synthetic.cpp
    src/util/assimp/helper.cpp
    src/util/IDGen.cpp
    src/util/exceptions.cpp
    # # Noise
    # src/rmagine/noise/noise.cpp
    src/noise/Noise.cpp
    src/noise/GaussianNoise.cpp
    src/noise/RelGaussianNoise.cpp
    src/noise/UniformDustNoise.cpp
)

add_library(rmagine-core SHARED
    ${RMAGINE_CORE_SRCS}
)

target_include_directories(rmagine-core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(rmagine-core
    ${ASSIMP_LIBRARIES}
    Eigen3::Eigen
)

target_compile_features(rmagine-core PRIVATE cxx_std_17)

set_target_properties(rmagine-core
  PROPERTIES
    EXPORT_NAME core
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

# TODO: do this:
add_library(rmagine ALIAS rmagine-core)
add_library(rmagine::core ALIAS rmagine-core)


set(RMAGINE_LIBRARIES rmagine-core)

add_library(rmagine-core-static STATIC
    ${RMAGINE_CORE_SRCS}
)

target_include_directories(rmagine-core-static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(rmagine-core-static
    PUBLIC
    ${ASSIMP_LIBRARIES}
    Eigen3::Eigen
)

target_compile_features(rmagine-core-static PRIVATE cxx_std_17)

set_target_properties(rmagine-core-static
  PROPERTIES
    EXPORT_NAME core-static
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::core-static ALIAS rmagine-core-static)

set(RMAGINE_STATIC_LIBRARIES rmagine-core-static)


# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

###########
## CORE
############
install(TARGETS rmagine-core EXPORT rmagine-core-targets)

install(EXPORT rmagine-core-targets
    FILE rmagine-core-targets.cmake
    COMPONENT core
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-core-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT core
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS rmagine-core-static EXPORT rmagine-core-static-targets)

install(EXPORT rmagine-core-static-targets
    FILE rmagine-core-static-targets.cmake
    COMPONENT core-static
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-static-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-core-static-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-static-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-static-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-static-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
  COMPONENT core-static
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT core-static
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)