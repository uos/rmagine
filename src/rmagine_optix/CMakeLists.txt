message(STATUS "Building OptiX (${OptiX_VERSION}) backend. Library: rmagine-optix")

# Optix Kernels
set(RMAGINE_OPTIX_PTX_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rmagine_optix_ptx")
set(RMAGINE_OPTIX_PTX_GLOB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/rmagine_optix_ptx")

set(CUDA_GENERATED_OUTPUT_DIR ${RMAGINE_OPTIX_PTX_DIR})
# message(STATUS "Writing Optix Kernels to ${RMAGINE_OPTIX_PTX_DIR}")

set(OPTIX_KERNEL_FILES
    # Simulation
    ## Spherical
    src/simulation/optix/SphereProgramRanges.cu
    ## Pinhole
    src/simulation/optix/PinholeProgramRanges.cu
    ## O1Dn
    src/simulation/optix/O1DnProgramRanges.cu
    ## OnDn
    src/simulation/optix/OnDnProgramRanges.cu
    ## NEW
    src/simulation/optix/SphereProgramGen.cu
    src/simulation/optix/PinholeProgramGen.cu
    src/simulation/optix/O1DnProgramGen.cu
    src/simulation/optix/OnDnProgramGen.cu
    src/simulation/optix/SensorProgramHit.cu
)

get_target_property(RMAGINE_CORE_INCLUDES rmagine-core INTERFACE_INCLUDE_DIRECTORIES)


# find_package(CUDA)
# cuda_include_directories(
#     ${RMAGINE_CORE_INCLUDES}
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
#     ${OptiX_INCLUDE_DIRS}
# )

# cuda_compile_ptx(RMAGINE_OPTIX_PTX_FILES
#     ${OPTIX_KERNEL_FILES}
# )

# set_property(TARGET myptx PROPERTY CUDA_PTX_COMPILATION ON)

if(CUDAToolkit_FOUND)

    add_library(rmagine_optix_ptx OBJECT
        ${OPTIX_KERNEL_FILES}
    )

    target_include_directories(rmagine_optix_ptx PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OptiX_INCLUDE_DIRS}
    )

    set_property(TARGET rmagine_optix_ptx PROPERTY CUDA_PTX_COMPILATION ON)

    target_link_libraries(rmagine_optix_ptx
        ${OptiX_LIBRARIES}
        rmagine-core
        rmagine-cuda
    )

    message(STATUS "Generated some PTX files: rmagine_optix_ptx")

    # find_package(CUDA)
    # cuda_include_directories(
    #     ${RMAGINE_CORE_INCLUDES}
    #     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    #     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    #     ${OptiX_INCLUDE_DIRS}
    # )

    # cuda_compile_ptx(RMAGINE_OPTIX_PTX_FILES
    #     ${OPTIX_KERNEL_FILES}
    # )

else(CUDAToolkit_FOUND)

    cuda_include_directories(
        ${RMAGINE_CORE_INCLUDES}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${OptiX_INCLUDE_DIRS}
    )

    cuda_compile_ptx(RMAGINE_OPTIX_PTX_FILES
        ${OPTIX_KERNEL_FILES}
    )

    add_custom_target(rmagine_optix_ptx ALL
    DEPENDS rmagine-core ${RMAGINE_OPTIX_PTX_FILES} ${OPTIX_KERNEL_FILES}
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

endif(CUDAToolkit_FOUND)





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

target_compile_definitions(rmagine-optix PRIVATE 
    RMAGINE_OPTIX_PTX_GLOB_DIR="${RMAGINE_OPTIX_PTX_GLOB_DIR}"
    RMAGINE_OPTIX_PTX_DIR="${RMAGINE_OPTIX_PTX_DIR}"
)

target_include_directories(rmagine-optix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
)

# TODO: make this PRIVATE as soon as RMCL is not compiling OptiX code
target_include_directories(rmagine-optix PUBLIC
    ${OptiX_INCLUDE_DIRS}
)

# target_include_directories(rmagine-optix PRIVATE
#     ${OptiX_INCLUDE_DIRS}
# )

# Must be public, since headers contain optionals
# target_compile_features(rmagine-optix PUBLIC cxx_std_17)

target_link_libraries(rmagine-optix
    ${OptiX_LIBRARIES}
    rmagine-cuda
)

# target_link_libraries(rmagine-optix PRIVATE
#     ${OptiX_LIBRARIES}
#     rmagine-cuda
# )


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
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

include(CMakePackageConfigHelpers)

###########
## OptiX
############
install(TARGETS rmagine-optix
    EXPORT rmagine-optix-targets
    COMPONENT optix
)

install(EXPORT rmagine-optix-targets
    FILE rmagine-optix-targets.cmake
    COMPONENT optix
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-optix-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

# FIND PACKAGE FILES
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-optix-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT optix
)

# HEADERS
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT optix
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)


set(CPACK_DEBIAN_OPTIX_PACKAGE_DEPENDS "rmagine-cuda" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL optix)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)