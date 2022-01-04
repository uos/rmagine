cmake_minimum_required(VERSION 3.2)
project(imagine 
    LANGUAGES CXX C
    VERSION 1)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_TOOLS "Build tools" OFF)

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

include(GNUInstallDirs)

set(CMAKE_MODULE_PATH
  ${imagine_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH}
)

################################
## REQUIRED DEPS


############################
##    Eigen: Math         ##
############################
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

############################
## EMBREE: CPU Raycasting ##
############################
find_package(embree 3 REQUIRED)
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

add_library(imagine SHARED
    # Maps
    src/imagine/map/AssimpMap.cpp
    src/imagine/map/EmbreeMap.cpp
    # Math
    # Simulators
    src/imagine/simulation/SphereSimulatorEmbree.cpp
    src/imagine/simulation/PinholeSimulatorEmbree.cpp
    src/imagine/simulation/O1DnSimulatorEmbree.cpp
    src/imagine/simulation/OnDnSimulatorEmbree.cpp
    # Types
    src/imagine/types/Memory.cpp
    # Util
    src/imagine/util/StopWatch.cpp
)

target_link_libraries(imagine
  ${ASSIMP_LIBRARIES}
  ${embree_LIBRARY}
  Eigen3::Eigen)

target_compile_features(imagine PRIVATE cxx_std_17)

if(CUDA_FOUND)

add_library(imagine_cuda SHARED
    src/imagine/types/MemoryCuda.cu
    src/imagine/math/math.cu
    src/imagine/util/cuda/CudaContext.cpp
)

target_link_libraries(imagine_cuda
    imagine
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)

add_dependencies(imagine_cuda
    imagine
)

if(OptiX_FOUND)
    # Optix Kernels
    set(IMAGINE_OPTIX_PTX_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/imagine_optix_ptx")
    set(IMAGINE_OPTIX_PTX_GLOB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/imagine_optix_ptx")

    set(CUDA_GENERATED_OUTPUT_DIR ${IMAGINE_OPTIX_PTX_DIR})
    add_definitions( -DWITH_OPTIX )
    add_definitions( -DIMAGINE_OPTIX_PTX_DIR="${IMAGINE_OPTIX_PTX_DIR}" )
    add_definitions( -DIMAGINE_OPTIX_PTX_GLOB_DIR="${IMAGINE_OPTIX_PTX_GLOB_DIR}")

    message(STATUS "Writing Optix Kernels to ${IMAGINE_OPTIX_PTX_DIR}")

    set(OPTIX_KERNEL_FILES
        # Simulation
        ## Spherical
        src/imagine/simulation/optix/SphereProgramRanges.cu
        src/imagine/simulation/optix/SphereProgramNormals.cu
        src/imagine/simulation/optix/SphereProgramGeneric.cu
        ## Pinhole
        src/imagine/simulation/optix/PinholeProgramRanges.cu
        src/imagine/simulation/optix/PinholeProgramNormals.cu
        src/imagine/simulation/optix/PinholeProgramGeneric.cu
        ## O1Dn
        src/imagine/simulation/optix/O1DnProgramRanges.cu
        src/imagine/simulation/optix/O1DnProgramNormals.cu
        src/imagine/simulation/optix/O1DnProgramGeneric.cu
        ## OnDn
        src/imagine/simulation/optix/OnDnProgramRanges.cu
        src/imagine/simulation/optix/OnDnProgramNormals.cu
        src/imagine/simulation/optix/OnDnProgramGeneric.cu
    )

    cuda_compile_ptx(IMAGINE_OPTIX_PTX_FILES
        ${OPTIX_KERNEL_FILES}
    )

    message(STATUS "IMAGINE_OPTIX_PTX_FILES: ${IMAGINE_OPTIX_PTX_FILES}")

    add_custom_target(imagine_optix_ptx ALL
        DEPENDS ${IMAGINE_OPTIX_PTX_FILES} ${OPTIX_KERNEL_FILES}
        SOURCES ${OPTIX_KERNEL_FILES}
        VERBATIM)

    message(STATUS "CMAKE_SOURCE_DIR: ${imagine_SOURCE_DIR}")

    add_custom_command(
        TARGET imagine_optix_ptx POST_BUILD
        COMMAND ${CMAKE_COMMAND} 
            -DIMAGINE_SOURCE_DIR=${imagine_SOURCE_DIR} 
            -DIMAGINE_OPTIX_PTX_DIR=${IMAGINE_OPTIX_PTX_DIR} 
            -DOPTIX_KERNEL_FILES="${OPTIX_KERNEL_FILES}" 
            -P "${CMAKE_CURRENT_LIST_DIR}/cmake/CompileOptixKernels.cmake"
    )
    
    # OptiX GENERAL LIBRARY
    add_library(imagine_optix SHARED
        # Util 
        src/imagine/util/optix/OptixFunctionTable.cpp
        src/imagine/util/optix/OptixProgram.cpp
        src/imagine/util/optix/OptixUtil.cpp
        # Maps
        src/imagine/map/OptixMap.cpp
        # Simulation
        src/imagine/simulation/SphereSimulatorOptix.cpp
        src/imagine/simulation/PinholeSimulatorOptix.cpp
        src/imagine/simulation/O1DnSimulatorOptix.cpp
        src/imagine/simulation/OnDnSimulatorOptix.cpp
        # Simulation Programs
        ## Spherical
        src/imagine/simulation/optix/SphereProgramRanges.cpp
        src/imagine/simulation/optix/SphereProgramNormals.cpp
        src/imagine/simulation/optix/SphereProgramGeneric.cpp
        ## Pinhole
        src/imagine/simulation/optix/PinholeProgramRanges.cpp
        src/imagine/simulation/optix/PinholeProgramNormals.cpp
        src/imagine/simulation/optix/PinholeProgramGeneric.cpp
        ## O1Dn
        src/imagine/simulation/optix/O1DnProgramRanges.cpp
        src/imagine/simulation/optix/O1DnProgramNormals.cpp
        src/imagine/simulation/optix/O1DnProgramGeneric.cpp
        ## OnDn
        src/imagine/simulation/optix/OnDnProgramRanges.cpp
        src/imagine/simulation/optix/OnDnProgramNormals.cpp
        src/imagine/simulation/optix/OnDnProgramGeneric.cpp
    )

    target_include_directories(imagine_optix PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    )

    target_link_libraries(imagine_optix
        ${OptiX_LIBRARIES}
        imagine_cuda
    )

    add_dependencies(imagine_optix
        imagine_optix_ptx
        imagine_cuda
    )
    

endif(OptiX_FOUND)

endif(CUDA_FOUND)

#### TESTS
if(BUILD_TESTS)

add_subdirectory(src/imagine_tests/memory)
add_subdirectory(src/imagine_tests/embree_sim)
add_subdirectory(src/imagine_tests/basic_math)
add_subdirectory(src/imagine_tests/generic_model_sim)

if(CUDA_FOUND)

add_subdirectory(src/imagine_tests/cuda_math)

if(OptiX_FOUND)
add_subdirectory(src/imagine_tests/optix_sim)
add_subdirectory(src/imagine_tests/generic_sim)
endif(OptiX_FOUND)

endif(CUDA_FOUND)
endif(BUILD_TESTS)

#### TOOLS
add_subdirectory(src/imagine_tools/imagine_benchmark)

#### INSTALL
# 1. Headers
install(DIRECTORY include/imagine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# 2. Libraries
install(TARGETS imagine imagine_cuda imagine_optix)
# TODO: Components?
# install(TARGETS imagine COMPONENT core)
# install(TARGETS imagine_cuda COMPONENT cuda)
# install(TARGETS imagine_optix COMPONENT optix)

# Executables
install(TARGETS imagine_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

configure_package_config_file(imagineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/imagineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/imagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/imagineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/imagineConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/imagineConfigVersion.cmake
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/imagine )

# Uninstall
add_custom_target(uninstall COMMENT "Uninstall installed files")
add_custom_command(
    TARGET uninstall
    POST_BUILD
    COMMENT "Uninstall files with install_manifest.txt"
    COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
            install_manifest.txt to be uninstalled!
)
