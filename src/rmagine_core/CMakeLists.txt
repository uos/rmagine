
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
    src/types/mesh_types.cpp
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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
    ${ASSIMP_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(rmagine-core
    ${ASSIMP_LIBRARIES}
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)

set_target_properties(rmagine-core
  PROPERTIES
    EXPORT_NAME core
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
    COMPONENT core
    CXX_STANDARD 17
)

# TODO: do this:
add_library(rmagine ALIAS rmagine-core)
add_library(rmagine::core ALIAS rmagine-core)


list(APPEND RMAGINE_LIBRARIES rmagine-core)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)


# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

###########
## CORE
############
install(TARGETS rmagine-core 
  EXPORT rmagine-core-targets
  COMPONENT core)

install(EXPORT rmagine-core-targets
    FILE rmagine-core-targets.cmake
    COMPONENT core
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config-version.cmake
  VERSION ${rmagine_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-core-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-core-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
  COMPONENT core
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT core
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libeigen3-dev, libassimp-dev" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL core)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)

