cmake_minimum_required(VERSION 3.4...10.0)
project(rmagine 
    LANGUAGES CXX C
    VERSION 2.1.2)

option(BUILD_TOOLS "Build tools" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_DOCS "Build documentation" OFF)

include(GNUInstallDirs)

# message(STATUS "RMAGINE_VERSION: ${rmagine_VERSION}")

set(rmagine_DEFINITIONS "-DRMAGINE_VERSION=\"${rmagine_VERSION}\"")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_MAJOR=${rmagine_VERSION_MAJOR}")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_MINOR=${rmagine_VERSION_MINOR}")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_PATCH=${rmagine_VERSION_PATCH}")


## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "CMake Version: ${CMAKE_VERSION}")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    message(STATUS ">= 3.9 - Enabling Link Time Optimization")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# DEFAULT RELEASE
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS On)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_TMP_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)

set(CMAKE_MODULE_PATH
  ${rmagine_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH}
)

################################
## REQUIRED DEPS

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

############################
## EMBREE: CPU Raycasting ##
############################
# supported VERSIONS: Embree 3 and 4

find_package(embree 4)
if(NOT embree_FOUND)
    find_package(embree 3)
endif()

if(embree_FOUND)
    set(RMAGINE_EMBREE_VERSION_MAJOR ${embree_VERSION_MAJOR})
    list(APPEND rmagine_DEFINITIONS "-DRMAGINE_EMBREE_VERSION_MAJOR=${embree_VERSION_MAJOR}")
    # add_definitions(-DRMAGINE_EMBREE_VERSION_MAJOR=${embree_VERSION_MAJOR})

    # NO INCLUDE?
    if(NOT embree_LIBRARY)
        if(EMBREE_LIBRARY)
            set(embree_LIBRARY ${EMBREE_LIBRARY})
        elseif(Embree_LIBRARY)
            set(embree_LIBRARY ${Embree_LIBRARY})
        else()
            set(embree_LIBRARY embree)
        endif()
    endif()
else()
    message(FATAL "Embree not found")
endif(embree_FOUND)

####################
## ASSIMP: MeshIO ##
####################
find_package(assimp REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

#######################
## OpenMP: Threading ##
#######################
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

########################################
## Optional Deps

######################################
## CUDA: For Optix                  ##
######################################

include(CheckLanguage)

find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

######################################
## OptiX: RTX Raycasting ##
######################################
find_package(OptiX)
if(OptiX_FOUND)
    include_directories(${OptiX_INCLUDE_DIRS})
    list(APPEND rmagine_DEFINITIONS "-DRMAGINE_WITH_OPTIX")
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
)


add_definitions(${rmagine_DEFINITIONS})


message(STATUS "Building Core. Library: rmagine")

set(RMAGINE_CORE_SRCS
    # Maps
    src/rmagine/map/AssimpIO.cpp
    # # Math
    src/rmagine/math/math.cpp
    src/rmagine/math/linalg.cpp
    src/rmagine/math/SVD.cpp
    # Types
    src/rmagine/types/Memory.cpp
    src/rmagine/types/conversions.cpp
    src/rmagine/types/sensors.cpp
    # Util
    src/rmagine/util/synthetic.cpp
    src/rmagine/util/assimp/helper.cpp
    src/rmagine/util/IDGen.cpp
    src/rmagine/util/exceptions.cpp
    # # Noise
    # src/rmagine/noise/noise.cpp
    src/rmagine/noise/Noise.cpp
    src/rmagine/noise/GaussianNoise.cpp
    src/rmagine/noise/RelGaussianNoise.cpp
    src/rmagine/noise/UniformDustNoise.cpp
)

add_library(rmagine SHARED
    ${RMAGINE_CORE_SRCS}
)

target_link_libraries(rmagine
  ${ASSIMP_LIBRARIES}
  Eigen3::Eigen
)

target_compile_features(rmagine PRIVATE cxx_std_17)

set(RMAGINE_LIBRARIES rmagine)

add_library(rmagine_static STATIC
    ${RMAGINE_CORE_SRCS}
)

target_link_libraries(rmagine_static
  ${ASSIMP_LIBRARIES}
  Eigen3::Eigen
)

target_compile_features(rmagine_static PRIVATE cxx_std_17)

set(RMAGINE_STATIC_LIBRARIES rmagine_static)

### RMAGINE EMBREE LIB
if(embree_FOUND)
    message(STATUS "Building Embree (${embree_VERSION}) backend. Library: rmagine_embree")

    set(RMAGINE_EMBREE_SRCS
        # Maps
        src/rmagine/map/embree/EmbreeDevice.cpp
        src/rmagine/map/embree/EmbreeGeometry.cpp
        src/rmagine/map/embree/EmbreeMesh.cpp
        src/rmagine/map/embree/EmbreeScene.cpp
        src/rmagine/map/embree/EmbreeInstance.cpp
        src/rmagine/map/embree/EmbreePoints.cpp
        src/rmagine/map/embree/embree_shapes.cpp
        src/rmagine/map/EmbreeMap.cpp
        
        # Simulators
        src/rmagine/simulation/SphereSimulatorEmbree.cpp
        src/rmagine/simulation/PinholeSimulatorEmbree.cpp
        src/rmagine/simulation/O1DnSimulatorEmbree.cpp
        src/rmagine/simulation/OnDnSimulatorEmbree.cpp
    )

    add_library(rmagine_embree SHARED
        ${RMAGINE_EMBREE_SRCS}
    )

    target_link_libraries(rmagine_embree
        rmagine
        ${embree_LIBRARY}
    )

    add_dependencies(rmagine_embree
        rmagine
    )

    target_compile_features(rmagine_embree PRIVATE cxx_std_17)

    list(APPEND RMAGINE_LIBRARIES rmagine_embree)


    add_library(rmagine_embree_static STATIC
        ${RMAGINE_EMBREE_SRCS}
    )

    target_link_libraries(rmagine_embree_static
        rmagine_static
        ${embree_LIBRARY}
    )

    add_dependencies(rmagine_embree_static
        rmagine_static
    )

    target_compile_features(rmagine_embree_static PRIVATE cxx_std_17)

    list(APPEND RMAGINE_STATIC_LIBRARIES rmagine_embree_static)

endif(embree_FOUND)

if(CUDA_FOUND)

    message(STATUS "Building Cuda dependend code. Library: rmagine_cuda")

    set(RMAGINE_CUDA_SRCS
        src/rmagine/types/MemoryCuda.cu
        # math
        src/rmagine/math/math.cu
        src/rmagine/math/math_batched.cu
        src/rmagine/math/SVDCuda.cpp
        # util
        src/rmagine/util/cuda/CudaContext.cpp
        src/rmagine/util/cuda/CudaDebug.cpp
        src/rmagine/util/cuda/CudaStream.cpp
        src/rmagine/util/cuda/random.cu
        # noise
        src/rmagine/noise/NoiseCuda.cu
        src/rmagine/noise/GaussianNoiseCuda.cu
        src/rmagine/noise/RelGaussianNoiseCuda.cu
        src/rmagine/noise/UniformDustNoiseCuda.cu
        # map
        src/rmagine/map/mesh_preprocessing.cu
    )

    add_library(rmagine_cuda SHARED
        ${RMAGINE_CUDA_SRCS}
    )

    # TODO: use CUDA language instead of find_package
    target_link_libraries(rmagine_cuda
        rmagine
        ${CUDA_LIBRARIES}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cublas_LIBRARY}
        cuda
    )

    add_dependencies(rmagine_cuda
        rmagine
    )

    list(APPEND RMAGINE_LIBRARIES rmagine_cuda)

    add_library(rmagine_cuda_static STATIC
        ${RMAGINE_CUDA_SRCS}
    )

    # TODO: use CUDA language instead of find_package
    target_link_libraries(rmagine_cuda_static
        rmagine_static
        ${CUDA_LIBRARIES}
        ${CUDA_cusolver_LIBRARY}
        ${CUDA_cublas_LIBRARY}
        cuda
    )

    add_dependencies(rmagine_cuda_static
        rmagine_static
    )

    list(APPEND RMAGINE_STATIC_LIBRARIES rmagine_cuda_static)


    if(OptiX_FOUND)

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
            src/rmagine/simulation/optix/SphereProgramRanges.cu
            # src/rmagine/simulation/optix/SphereProgramNormals.cu
            # src/rmagine/simulation/optix/SphereProgramGeneric.cu
            ## Pinhole
            src/rmagine/simulation/optix/PinholeProgramRanges.cu
            # src/rmagine/simulation/optix/PinholeProgramNormals.cu
            # src/rmagine/simulation/optix/PinholeProgramGeneric.cu
            # ## O1Dn
            src/rmagine/simulation/optix/O1DnProgramRanges.cu
            # src/rmagine/simulation/optix/O1DnProgramNormals.cu
            # src/rmagine/simulation/optix/O1DnProgramGeneric.cu
            # ## OnDn
            src/rmagine/simulation/optix/OnDnProgramRanges.cu
            # src/rmagine/simulation/optix/OnDnProgramNormals.cu
            # src/rmagine/simulation/optix/OnDnProgramGeneric.cu
            ## NEW
            src/rmagine/simulation/optix/SphereProgramGen.cu
            src/rmagine/simulation/optix/PinholeProgramGen.cu
            src/rmagine/simulation/optix/O1DnProgramGen.cu
            src/rmagine/simulation/optix/OnDnProgramGen.cu
            src/rmagine/simulation/optix/SensorProgramHit.cu
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
            src/rmagine/util/optix/OptixFunctionTable.cpp
            src/rmagine/util/optix/OptixUtil.cpp
            src/rmagine/util/optix/OptixContext.cpp
            src/rmagine/util/optix/optix_modules.cpp
            # Maps
            src/rmagine/map/optix/OptixAccelerationStructure.cpp
            src/rmagine/map/optix/OptixGeometry.cpp
            src/rmagine/map/optix/OptixMesh.cpp
            src/rmagine/map/optix/OptixInst.cpp
            src/rmagine/map/optix/OptixScene.cpp
            src/rmagine/map/optix/OptixTransformable.cpp
            src/rmagine/map/optix/OptixEntity.cpp
            src/rmagine/map/optix/OptixSceneEventReceiver.cpp
            src/rmagine/map/optix/optix_shapes.cpp
            src/rmagine/map/OptixMap.cpp
            # Simulation
            src/rmagine/simulation/SphereSimulatorOptix.cpp
            src/rmagine/simulation/PinholeSimulatorOptix.cpp
            src/rmagine/simulation/O1DnSimulatorOptix.cpp
            src/rmagine/simulation/OnDnSimulatorOptix.cpp
            # Simulation Programs
            src/rmagine/simulation/optix/sim_modules.cpp
            src/rmagine/simulation/optix/sim_program_groups.cpp
            src/rmagine/simulation/optix/sim_pipelines.cpp
        )

        # OptiX GENERAL LIBRARY
        add_library(rmagine_optix SHARED
            ${RMAGINE_OPTIX_SRCS}
        )

        target_include_directories(rmagine_optix PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        )

        target_link_libraries(rmagine_optix
            ${OptiX_LIBRARIES}
            rmagine_cuda
        )

        add_dependencies(rmagine_optix
            rmagine_optix_ptx
            rmagine_cuda
        )

        list(APPEND RMAGINE_LIBRARIES rmagine_optix)

        # OptiX GENERAL STATIC LIBRARY
        add_library(rmagine_optix_static STATIC
            ${RMAGINE_OPTIX_SRCS}
        )

        target_include_directories(rmagine_optix_static PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        )

        target_link_libraries(rmagine_optix_static
            ${OptiX_LIBRARIES}
            rmagine_cuda_static
        )

        add_dependencies(rmagine_optix
            rmagine_optix_ptx
            rmagine_cuda_static
        )

        list(APPEND RMAGINE_STATIC_LIBRARIES rmagine_optix_static)

    endif(OptiX_FOUND)

endif(CUDA_FOUND)

message(STATUS "RMAGINE_LIBRARIES: ${RMAGINE_LIBRARIES}")

#### TESTS
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTS)

#### TOOLS
if(BUILD_TOOLS)
    add_subdirectory(src/rmagine_tools/rmagine_benchmark)
    add_subdirectory(src/rmagine_tools/rmagine_synthetic)
    add_subdirectory(src/rmagine_tools/rmagine_map_info)
    add_subdirectory(src/rmagine_tools/rmagine_version)
endif(BUILD_TOOLS)

if(BUILD_EXAMPLES)
    if(embree_FOUND)
        add_subdirectory(src/rmagine_examples/rmagine_embree_scene)
    endif(embree_FOUND)

    if(OptiX_FOUND)
    add_subdirectory(src/rmagine_examples/rmagine_optix_scene)
    endif(OptiX_FOUND)
endif(BUILD_EXAMPLES)


#### INSTALL
# 1. Headers
install(DIRECTORY include/rmagine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# 2. Libraries
install(TARGETS ${RMAGINE_LIBRARIES} EXPORT rmagineTargets)
# 3. Static Libraries
install(TARGETS ${RMAGINE_STATIC_LIBRARIES} EXPORT rmagineTargets)



install(EXPORT rmagineTargets
    FILE rmagineTargets.cmake
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine)




# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/rmagineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine)

# Shortcut Headers
include(HeaderFromDirectory)
GenerateHeadersFromDirectories(
    ${rmagine_SOURCE_DIR}/include/rmagine 
    ${CMAKE_TMP_OUTPUT_DIRECTORY} 
    GEN_SHORTCUT_HEADERS)

message(STATUS "Generated ${GEN_SHORTCUT_HEADERS}")
install(FILES ${GEN_SHORTCUT_HEADERS} 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine )



# Uninstall
# https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rmagineUninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rmagineUninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/rmagineUninstall.cmake
    COMMENT "Uninstall rmagine libraries and all header files")
endif()


###############################################################################
# DOCUMENTATION
###############################################################################

if(BUILD_DOCS)

find_package( Doxygen REQUIRED)

message(STATUS "Build Documentation with 'make doc'")

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

add_custom_target( doc ALL 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM )

endif(BUILD_DOCS)

