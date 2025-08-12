##########################
## Package: rmagine     ##
## Component: optix      ##
##########################

##################################
## Required external libraries: ##
## - CUDA                       ##
## - OptiX                      ##
##################################

set(RMAGINE_OPTIX_DOWNLOAD_VERSION 9.0.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# I am aware find_package(CUDA) is deprecated.
# - TODO: support find_package(CUDAToolkit) as well
# -- Example: https://github.com/ceres-solver/ceres-solver/blob/master/CMakeLists.txt
# - TODO: this becomes quite messy. make a seperate file from this

# default flags are not set when including cuda
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)

  if(NOT CUDA_NVCC_FLAGS_DEBUG)
    set(CUDA_NVCC_FLAGS_DEBUG "-g" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_MINSIZEREL)
    set(CUDA_NVCC_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_RELEASE)
    set(CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_RELWITHDEBINFO)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "" FORCE)
  endif()

endif()

# set(CUDA_STANDARD 14)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CUDA_FOUND True)
        set(CUDA_LIBRARIES CUDA::cudart)
        set(CUDA_cusolver_LIBRARY CUDA::cusolver)
        set(CUDA_cublas_LIBRARY CUDA::cublas)
        set(CUDA_DRIVER_LIBRARY CUDA::cuda_driver)
        set(CUDA_INCLUDE_DIRS "") # is in target instead
    else()
        find_package(CUDA)
        if(CUDA_FOUND)
            enable_language(CUDA)
            set(CUDA_DRIVER_LIBRARY cuda)
        else()
            message(STATUS "Neither CudaToolkit nor CUDA found!")
        endif(CUDA_FOUND)
    endif(CUDAToolkit_FOUND)
endif(CMAKE_CUDA_COMPILER)

if(CUDA_FOUND)

set(TMP_LIST_AUTO_LOCAL "Auto;Local")
if(${RMAGINE_CPM_MODE} IN_LIST TMP_LIST_AUTO_LOCAL)
    find_package(OptiX QUIET)
endif()

set(TMP_LIST_AUTO_DOWNLOAD "Auto;Download")
if(NOT OptiX_FOUND AND ${RMAGINE_CPM_MODE} IN_LIST TMP_LIST_AUTO_DOWNLOAD)

    # TODO: Download
    CPMAddPackage(
            NAME optix
            GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
            VERSION ${RMAGINE_OPTIX_DOWNLOAD_VERSION}
            DOWNLOAD_ONLY YES
        )

    if(optix_ADDED)
        set(OptiX_INCLUDE_DIRS ${optix_SOURCE_DIR}/include)
        set(OptiX_FOUND TRUE)
        set(OptiX_VERSION ${RMAGINE_OPTIX_DOWNLOAD_VERSION})
    endif(optix_ADDED)
endif()

if(OptiX_FOUND)
    list(APPEND rmagine_DEFINITIONS "-DRMAGINE_WITH_OPTIX")
endif(OptiX_FOUND)

endif(CUDA_FOUND)


if(OptiX_FOUND)

message(STATUS "${Green}rmagine::optix${ColourReset}: Building (OptiX v${OptiX_VERSION})")

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

if(CUDAToolkit_FOUND)
    # NEW VERSION

    add_library(rmagine_optix_cu_to_ptx OBJECT
        ${OPTIX_KERNEL_FILES}
    )

    target_include_directories(rmagine_optix_cu_to_ptx PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OptiX_INCLUDE_DIRS}
    )

    set_target_properties(rmagine_optix_cu_to_ptx 
        PROPERTIES
            CUDA_PTX_COMPILATION ON
            # CUDA_ARCHITECTURES all
    )

    
    # if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.23" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    #     set_target_properties(rmagine_optix_cu_to_ptx PROPERTIES CUDA_ARCHITECTURES all)
    # endif()
    # -> nvcc fatal   : Option '--ptx (-ptx)' is not allowed when compiling for multiple GPU architectures

    target_link_libraries(rmagine_optix_cu_to_ptx
        ${OptiX_LIBRARIES}
        rmagine-core
        rmagine-cuda
    )

    add_custom_target(rmagine_optix_ptx ALL
        DEPENDS rmagine-core rmagine_optix_cu_to_ptx ${OPTIX_KERNEL_FILES}
        SOURCES ${OPTIX_KERNEL_FILES}
        VERBATIM)

    add_custom_command(
        TARGET rmagine_optix_ptx POST_BUILD
        COMMAND ${CMAKE_COMMAND} 
            -DRMAGINE_SOURCE_DIR=${rmagine_SOURCE_DIR} 
            -DRMAGINE_OPTIX_PTX_DIR=${RMAGINE_OPTIX_PTX_DIR}
            -DOPTIX_KERNEL_FILES="${OPTIX_KERNEL_FILES}"
            -P "${CMAKE_CURRENT_LIST_DIR}/cmake/CompileOptixKernelsCudaToolkit.cmake"
    )

else(CUDAToolkit_FOUND)
    # THIS IS GOING TO BE OBSOLETE

    cuda_include_directories(
        ${RMAGINE_CORE_INCLUDES}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
    src/simulation/SimulatorOptix.cpp
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
target_include_directories(rmagine-optix PRIVATE
    ${OptiX_INCLUDE_DIRS}
)

# Must be public, since headers contain optionals
# target_compile_features(rmagine-optix PUBLIC cxx_std_17)
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

# set CUDA_ARCHITECTURES with CMake >= 3.23 when CMAKE_* is not set
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.23" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(rmagine-optix PROPERTIES CUDA_ARCHITECTURES all)
endif()

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

else(OptiX_FOUND)

message(STATUS "${Yellow}rmagine::optix${ColourReset}: Skipping. Missing Dependencies.")

endif(OptiX_FOUND)