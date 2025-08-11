##########################
## Package: rmagine     ##
## Component: cuda      ##
##########################

##################################
## Required external libraries: ##
## - CUDA                       ##
##################################

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
        set(CUDA_VERSION ${CUDAToolkit_VERSION})
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

message(STATUS "${Green}rmagine::cuda${ColourReset}: Building (CUDA v${CUDA_VERSION})")

set(RMAGINE_CUDA_SRCS
    src/types/MemoryCuda.cu
    # math
    src/math/memory_math.cu
    src/math/math_batched.cu
    src/math/linalg.cu
    src/math/statistics.cu
    # util
    src/util/cuda/CudaContext.cpp
    src/util/cuda/CudaDebug.cpp
    src/util/cuda/CudaStream.cpp
    src/util/cuda/random.cu
    # noise
    src/noise/NoiseCuda.cu
    src/noise/GaussianNoiseCuda.cu
    src/noise/RelGaussianNoiseCuda.cu
    src/noise/UniformDustNoiseCuda.cu
    # map
    src/map/mesh_preprocessing.cu
)

add_library(rmagine-cuda SHARED
    ${RMAGINE_CUDA_SRCS}
)

target_include_directories(rmagine-cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
    ${CUDA_INCLUDE_DIRS}
)

# TODO: use CUDA language instead of find_package
target_link_libraries(rmagine-cuda PUBLIC
    rmagine-core
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
)

add_dependencies(rmagine-cuda
    rmagine-core
)

# enable_language(CUDA)
# target_compile_features(rmagine-cuda PUBLIC cxx_std_17 cuda_std_17)

# Problem: We need CUDA C++17 support, since some template functions 
# have constexpr etc. Ubuntu 20s standard cmake (version 3.16) does not support 
# to set the cuda standard higher than 14. This is not a clean solution.
# Clean solution:
# 1. Make a warning for Ubuntu 20 / cmake 3.16 solution to compile at own risk.
# 2. Compile for newer Ubtunu/cmake versions with a proper target property
set_target_properties(rmagine-cuda
    PROPERTIES
        EXPORT_NAME cuda
        SOVERSION ${rmagine_VERSION_MAJOR}
        VERSION ${rmagine_VERSION}
        CUDA_SEPARABLE_COMPILATION ON
        # CUDA_STANDARD 17 -> only works for 
        # CXX_STANDARD 17
)

# set CUDA_ARCHITECTURES with CMake >= 3.23 when CMAKE_* is not set
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.23" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(rmagine-cuda PROPERTIES CUDA_ARCHITECTURES all)
endif()

add_library(rmagine::cuda ALIAS rmagine-cuda)

list(APPEND RMAGINE_LIBRARIES rmagine-cuda)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

###########
## INSTALL COMPONENT: CUDA
############
install(TARGETS rmagine-cuda
    EXPORT rmagine-cuda-targets
    COMPONENT cuda
)

install(EXPORT rmagine-cuda-targets
    FILE rmagine-cuda-targets.cmake
    COMPONENT cuda
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config-version.cmake
    VERSION ${rmagine_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-cuda-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

# FIND PACKAGE FILES
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT cuda
)

# HEADERS
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT cuda
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_CUDA_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL cuda)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)

else(CUDA_FOUND)

message(STATUS "${Yellow}rmagine::cuda${ColourReset}: Skipping. Missing Dependencies.")

endif(CUDA_FOUND)
