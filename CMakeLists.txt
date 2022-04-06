cmake_minimum_required(VERSION 3.2)
project(rmagine 
    LANGUAGES CXX C
    VERSION 1)

include(GNUInstallDirs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
#set(CMAKE_CXX_STANDARD 17)


option(BUILD_TOOLS "Build tools" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_DOCS "Build documentation" OFF)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

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

############################
## EMBREE: CPU Raycasting ##
############################
find_package(embree 3)
if(embree_FOUND)
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
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
)

set(RMAGINE_LIBRARIES rmagine)

add_library(rmagine SHARED
    # Maps
    src/rmagine/map/AssimpMap.cpp
    # Math
    src/rmagine/math/math.cpp
    # Types
    src/rmagine/types/Memory.cpp
    # Util
    src/rmagine/util/StopWatch.cpp
    src/rmagine/util/synthetic.cpp
    # Noise
    src/rmagine/noise/noise.cpp
)

target_link_libraries(rmagine
  ${ASSIMP_LIBRARIES})

target_compile_features(rmagine PRIVATE cxx_std_17)

### RMAGINE EMBREE LIB
if(embree_FOUND)
add_library(rmagine_embree SHARED
    # Maps
    src/rmagine/map/EmbreeMap.cpp
    # Simulators
    src/rmagine/simulation/SphereSimulatorEmbree.cpp
    src/rmagine/simulation/PinholeSimulatorEmbree.cpp
    src/rmagine/simulation/O1DnSimulatorEmbree.cpp
    src/rmagine/simulation/OnDnSimulatorEmbree.cpp
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

endif(embree_FOUND)

if(CUDA_FOUND)

add_library(rmagine_cuda SHARED
    src/rmagine/types/MemoryCuda.cu
    # math
    src/rmagine/math/math.cu
    src/rmagine/math/math_batched.cu
    src/rmagine/math/SVD_cuda.cpp
    # util
    src/rmagine/util/cuda/CudaContext.cpp
    # Noise
    src/rmagine/noise/noise.cu
)

target_link_libraries(rmagine_cuda
    rmagine
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
)

add_dependencies(rmagine_cuda
    rmagine
)

list(APPEND RMAGINE_LIBRARIES rmagine_cuda)

if(OptiX_FOUND)
    # Optix Kernels
    set(RMAGINE_OPTIX_PTX_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/rmagine_optix_ptx")
    set(RMAGINE_OPTIX_PTX_GLOB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/rmagine_optix_ptx")

    set(CUDA_GENERATED_OUTPUT_DIR ${RMAGINE_OPTIX_PTX_DIR})
    add_definitions( -DWITH_OPTIX )
    add_definitions( -DRMAGINE_OPTIX_PTX_DIR="${RMAGINE_OPTIX_PTX_DIR}" )
    add_definitions( -DRMAGINE_OPTIX_PTX_GLOB_DIR="${RMAGINE_OPTIX_PTX_GLOB_DIR}")

    message(STATUS "Writing Optix Kernels to ${RMAGINE_OPTIX_PTX_DIR}")

    set(OPTIX_KERNEL_FILES
        # Simulation
        ## Spherical
        src/rmagine/simulation/optix/SphereProgramRanges.cu
        src/rmagine/simulation/optix/SphereProgramNormals.cu
        src/rmagine/simulation/optix/SphereProgramGeneric.cu
        ## Pinhole
        src/rmagine/simulation/optix/PinholeProgramRanges.cu
        src/rmagine/simulation/optix/PinholeProgramNormals.cu
        src/rmagine/simulation/optix/PinholeProgramGeneric.cu
        ## O1Dn
        src/rmagine/simulation/optix/O1DnProgramRanges.cu
        src/rmagine/simulation/optix/O1DnProgramNormals.cu
        src/rmagine/simulation/optix/O1DnProgramGeneric.cu
        ## OnDn
        src/rmagine/simulation/optix/OnDnProgramRanges.cu
        src/rmagine/simulation/optix/OnDnProgramNormals.cu
        src/rmagine/simulation/optix/OnDnProgramGeneric.cu
    )

    cuda_compile_ptx(RMAGINE_OPTIX_PTX_FILES
        ${OPTIX_KERNEL_FILES}
    )

    add_custom_target(rmagine_optix_ptx ALL
        DEPENDS ${RMAGINE_OPTIX_PTX_FILES} ${OPTIX_KERNEL_FILES}
        SOURCES ${OPTIX_KERNEL_FILES}
        VERBATIM)

    message(STATUS "CMAKE_SOURCE_DIR: ${rmagine_SOURCE_DIR}")

    add_custom_command(
        TARGET rmagine_optix_ptx POST_BUILD
        COMMAND ${CMAKE_COMMAND} 
            -DRMAGINE_SOURCE_DIR=${rmagine_SOURCE_DIR} 
            -DRMAGINE_OPTIX_PTX_DIR=${RMAGINE_OPTIX_PTX_DIR} 
            -DOPTIX_KERNEL_FILES="${OPTIX_KERNEL_FILES}" 
            -P "${CMAKE_CURRENT_LIST_DIR}/cmake/CompileOptixKernels.cmake"
    )
    
    # OptiX GENERAL LIBRARY
    add_library(rmagine_optix SHARED
        # Util 
        src/rmagine/util/optix/OptixFunctionTable.cpp
        src/rmagine/util/optix/OptixProgram.cpp
        src/rmagine/util/optix/OptixUtil.cpp
        src/rmagine/util/optix/OptixContext.cpp
        # Maps
        src/rmagine/map/OptixMap.cpp
        # Simulation
        src/rmagine/simulation/SphereSimulatorOptix.cpp
        src/rmagine/simulation/PinholeSimulatorOptix.cpp
        src/rmagine/simulation/O1DnSimulatorOptix.cpp
        src/rmagine/simulation/OnDnSimulatorOptix.cpp
        # Simulation Programs
        ## Spherical
        src/rmagine/simulation/optix/SphereProgramRanges.cpp
        src/rmagine/simulation/optix/SphereProgramNormals.cpp
        src/rmagine/simulation/optix/SphereProgramGeneric.cpp
        ## Pinhole
        src/rmagine/simulation/optix/PinholeProgramRanges.cpp
        src/rmagine/simulation/optix/PinholeProgramNormals.cpp
        src/rmagine/simulation/optix/PinholeProgramGeneric.cpp
        ## O1Dn
        src/rmagine/simulation/optix/O1DnProgramRanges.cpp
        src/rmagine/simulation/optix/O1DnProgramNormals.cpp
        src/rmagine/simulation/optix/O1DnProgramGeneric.cpp
        ## OnDn
        src/rmagine/simulation/optix/OnDnProgramRanges.cpp
        src/rmagine/simulation/optix/OnDnProgramNormals.cpp
        src/rmagine/simulation/optix/OnDnProgramGeneric.cpp
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

endif(OptiX_FOUND)

endif(CUDA_FOUND)

#### TESTS
if(BUILD_TESTS)

## ADDITIONAL DEPS FOR TESTS
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(Eigen3_FOUND)

add_subdirectory(src/rmagine_tests/memory)
add_subdirectory(src/rmagine_tests/embree_sim)

if(Eigen3_FOUND)
    add_subdirectory(src/rmagine_tests/basic_math)
endif(Eigen3_FOUND)

add_subdirectory(src/rmagine_tests/generic_model_sim)
add_subdirectory(src/rmagine_tests/synthetic)

if(CUDA_FOUND)

add_subdirectory(src/rmagine_tests/cuda_math)
add_subdirectory(src/rmagine_tests/svd)

if(OptiX_FOUND)
add_subdirectory(src/rmagine_tests/optix_sim)
add_subdirectory(src/rmagine_tests/generic_sim)
endif(OptiX_FOUND)

endif(CUDA_FOUND)
endif(BUILD_TESTS)

#### TOOLS
if(BUILD_TOOLS)
add_subdirectory(src/rmagine_tools/rmagine_benchmark)
endif(BUILD_TOOLS)

#### INSTALL
# 1. Headers
install(DIRECTORY include/rmagine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# 2. Libraries
install(TARGETS ${RMAGINE_LIBRARIES})

# install(TARGETS rmagine rmagine_cuda rmagine_optix)
# TODO: Components?
# install(TARGETS rmagine COMPONENT core)
# install(TARGETS rmagine_cuda COMPONENT cuda)
# install(TARGETS rmagine_optix COMPONENT optix)

if(BUILD_TOOLS)
# Install Tools
install(TARGETS rmagine_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

endif(BUILD_TOOLS)


# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

configure_package_config_file(rmagineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine )

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
add_custom_target(uninstall COMMENT "Uninstall installed files")
add_custom_command(
    TARGET uninstall
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)

###############################################################################
# DOCUMENTATION
###############################################################################

if(BUILD_DOCS)

find_package( Doxygen REQUIRED)

message(STATUS "Build Documentation with 'make doc'")

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

add_custom_target( doc ALL 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM )

endif(BUILD_DOCS)

