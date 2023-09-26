message(STATUS "Building OptiX (${OptiX_VERSION}) backend. Library: rmagine_optix")

# Optix Kernels
set(RMAGINE_OPTIX_PTX_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rmagine_optix_ptx")
set(RMAGINE_OPTIX_PTX_GLOB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/rmagine_optix_ptx")

set(CUDA_GENERATED_OUTPUT_DIR ${RMAGINE_OPTIX_PTX_DIR})
add_definitions( -DRMAGINE_OPTIX_PTX_DIR="${RMAGINE_OPTIX_PTX_DIR}" )
add_definitions( -DRMAGINE_OPTIX_PTX_GLOB_DIR="${RMAGINE_OPTIX_PTX_GLOB_DIR}")

# message(STATUS "Writing Optix Kernels to ${RMAGINE_OPTIX_PTX_DIR}")

set(OPTIX_KERNEL_FILES
    # Simulation
    ## Spherical
    src/simulation/optix/SphereProgramRanges.cu
    ## Pinhole
    src/simulation/optix/PinholeProgramRanges.cu
    # ## O1Dn
    src/simulation/optix/O1DnProgramRanges.cu
    # ## OnDn
    src/simulation/optix/OnDnProgramRanges.cu
    ## NEW
    src/simulation/optix/SphereProgramGen.cu
    src/simulation/optix/PinholeProgramGen.cu
    src/simulation/optix/O1DnProgramGen.cu
    src/simulation/optix/OnDnProgramGen.cu
    src/simulation/optix/SensorProgramHit.cu
)

cuda_compile_ptx(RMAGINE_OPTIX_PTX_FILES
    ${OPTIX_KERNEL_FILES}
)

add_custom_target(rmagine_optix_ptx ALL
    DEPENDS ${RMAGINE_OPTIX_PTX_FILES} ${OPTIX_KERNEL_FILES}
    SOURCES ${OPTIX_KERNEL_FILES}
    VERBATIM)

add_custom_command(
    TARGET rmagine_optix_ptx POST_BUILD
    COMMAND ${CMAKE_COMMAND} 
        -DRMAGINE_SOURCE_DIR=${rmagine_SOURCE_DIR} 
        -DRMAGINE_OPTIX_PTX_DIR=${RMAGINE_OPTIX_PTX_DIR} 
        -DOPTIX_KERNEL_FILES="${OPTIX_KERNEL_FILES}" 
        -P "${CMAKE_CURRENT_LIST_DIR}/cmake/CompileOptixKernels.cmake"
)

set(RMAGINE_OPTIX_SRCS
    # Util 
    src/util/optix/OptixFunctionTable.cpp
    src/util/optix/OptixUtil.cpp
    src/util/optix/OptixContext.cpp
    src/util/optix/optix_modules.cpp
    # Maps
    src/map/optix/OptixAccelerationStructure.cpp
    src/map/optix/OptixGeometry.cpp
    src/map/optix/OptixMesh.cpp
    src/map/optix/OptixInst.cpp
    src/map/optix/OptixScene.cpp
    src/map/optix/OptixTransformable.cpp
    src/map/optix/OptixEntity.cpp
    src/map/optix/OptixSceneEventReceiver.cpp
    src/map/optix/optix_shapes.cpp
    src/map/OptixMap.cpp
    # Simulation
    src/simulation/SphereSimulatorOptix.cpp
    src/simulation/PinholeSimulatorOptix.cpp
    src/simulation/O1DnSimulatorOptix.cpp
    src/simulation/OnDnSimulatorOptix.cpp
    # Simulation Programs
    src/simulation/optix/sim_modules.cpp
    src/simulation/optix/sim_program_groups.cpp
    src/simulation/optix/sim_pipelines.cpp
)

# OptiX GENERAL LIBRARY
add_library(rmagine-optix SHARED
    ${RMAGINE_OPTIX_SRCS}
)

target_include_directories(rmagine-optix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rmagine-optix
    ${OptiX_LIBRARIES}
    rmagine-cuda
)

add_dependencies(rmagine-optix
    rmagine_optix_ptx
    rmagine-cuda
)

set_target_properties(rmagine-optix
  PROPERTIES
    EXPORT_NAME optix
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::optix ALIAS rmagine-optix)

list(APPEND RMAGINE_LIBRARIES rmagine-optix)

# OptiX GENERAL STATIC LIBRARY
add_library(rmagine-optix-static STATIC
    ${RMAGINE_OPTIX_SRCS}
)

target_include_directories(rmagine-optix-static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rmagine-optix-static
    ${OptiX_LIBRARIES}
    rmagine-cuda-static
)

add_dependencies(rmagine-optix
    rmagine_optix_ptx
    rmagine-cuda-static
)

set_target_properties(rmagine-optix-static
  PROPERTIES
    EXPORT_NAME optix-static
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::optix-static ALIAS rmagine-optix-static)

list(APPEND RMAGINE_STATIC_LIBRARIES rmagine-optix-static)

include(CMakePackageConfigHelpers)

###########
## OptiX
############
install(TARGETS rmagine-optix EXPORT rmagine-optix-targets)

install(EXPORT rmagine-optix-targets
    FILE rmagine-optix-targets.cmake
    COMPONENT optix
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-optix-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
    COMPONENT optix
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT optix
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS rmagine-optix-static EXPORT rmagine-optix-static-targets)

install(EXPORT rmagine-optix-static-targets
    FILE rmagine-optix-static-targets.cmake
    COMPONENT optix-static
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-static-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-optix-static-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-static-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-static-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-static-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
    COMPONENT optix-static
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT optix-static
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)