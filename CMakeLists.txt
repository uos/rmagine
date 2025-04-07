cmake_minimum_required(VERSION 3.16)
project(rmagine 
    LANGUAGES CXX C
    VERSION 2.2.12)# TODO update this version when merging into main-branch

option(BUILD_TOOLS "Build tools" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_DOCS "Build documentation" OFF)

include(GNUInstallDirs)

set(rmagine_DEFINITIONS "-DRMAGINE_VERSION=\"${rmagine_VERSION}\"")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_MAJOR=${rmagine_VERSION_MAJOR}")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_MINOR=${rmagine_VERSION_MINOR}")
list(APPEND rmagine_DEFINITIONS "-DRMAGINE_VERSION_PATCH=${rmagine_VERSION_PATCH}")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DEFAULT RELEASE
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_TMP_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tmp)

set(CMAKE_MODULE_PATH
  ${rmagine_SOURCE_DIR}/cmake
  ${CMAKE_MODULE_PATH}
)

# printing in color
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

################################
## REQUIRED DEPS

find_package(Boost CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

############################
## EMBREE: CPU Raycasting ##
############################
# supported Embree VERSIONS: Embree 4 (Tested until versio 4.2)
find_package(embree 4)

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
    message(STATUS "Embree not found")
endif(embree_FOUND)

####################
## ASSIMP: MeshIO ##
####################
find_package(assimp REQUIRED)

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

# for ouster config loading
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

if(TARGET PkgConfig::jsoncpp) # for downwards compatibility
    set(JSONCPP_INCLUDDE_DIRS "")
    set(JSONCPP_LIBRARIES PkgConfig::jsoncpp)
endif()

######################################
## CUDA: For Optix                  ##
######################################

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
    message(STATUS "Cuda language available!")

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "Found Cuda Toolkit!")
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
            message(STATUS "Found Cuda!")
            enable_language(CUDA)
            set(CUDA_DRIVER_LIBRARY cuda)
        else()
            message(STATUS "Neither CudaToolkit nor CUDA found!")
        endif(CUDA_FOUND)
    endif(CUDAToolkit_FOUND)
endif(CMAKE_CUDA_COMPILER)


######################################
## OptiX: RTX Raycasting ##
######################################
find_package(OptiX)
if(OptiX_FOUND)
    list(APPEND rmagine_DEFINITIONS "-DRMAGINE_WITH_OPTIX")
endif()

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
#     include
# )

add_definitions(${rmagine_DEFINITIONS})

set(RMAGINE_LIBRARIES)
set(RMAGINE_STATIC_LIBRARIES)

### RMAGINE CORE LIB
add_subdirectory(src/rmagine_core)

### RMAGINE OUSTER LIB
if(JSONCPP_FOUND)
    add_subdirectory(src/rmagine_ouster)
endif(JSONCPP_FOUND)

### RMAGINE EMBREE LIB
if(embree_FOUND)
  add_subdirectory(src/rmagine_embree)
endif(embree_FOUND)

### RMAGINE CUDA LIB
if(CUDA_FOUND)
    add_subdirectory(src/rmagine_cuda)
    if(OptiX_FOUND)
        add_subdirectory(src/rmagine_optix)
    endif(OptiX_FOUND)
endif(CUDA_FOUND)

message(STATUS "${BoldCyan}Components build:${ColourReset}")
foreach(LIBRARY ${RMAGINE_LIBRARIES})
    message(STATUS "- ${BoldGreen}${LIBRARY}${ColourReset}")
endforeach()

#### TESTS
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTS)

#### TOOLS
if(BUILD_TOOLS)
    add_subdirectory(apps/rmagine_benchmark)
    add_subdirectory(apps/rmagine_synthetic)
    add_subdirectory(apps/rmagine_map_info)
    add_subdirectory(apps/rmagine_version)
endif(BUILD_TOOLS)

if(BUILD_EXAMPLES)
if(embree_FOUND)
    add_subdirectory(apps/rmagine_examples/rmagine_embree_scene)
endif(embree_FOUND)

if(OptiX_FOUND)
    add_subdirectory(apps/rmagine_examples/rmagine_optix_scene)
endif(OptiX_FOUND)
endif(BUILD_EXAMPLES)


# CMAKE FIND SCRIPT
include(CMakePackageConfigHelpers)

#### INSTALL
# 1. Headers
# install(DIRECTORY include/rmagine 
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# 2. Libraries
# install(TARGETS rmagine-cuda rmagine-optix EXPORT rmagineTargets)
# 3. Static Libraries
# install(TARGETS rmagine-cuda-static rmagine-optix-static EXPORT rmagineTargets)

# install(EXPORT rmagineTargets
#     FILE rmagineTargets.cmake
#     NAMESPACE rmagine::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine)

    
# CMAKE FIND SCRIPT

configure_package_config_file(cmake/rmagine-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-config-version.cmake
    COMPATIBILITY SameMajorVersion
)

## INSTALL
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rmagine-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/rmagine-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT core
)

install(FILES package.xml
    DESTINATION share/rmagine-${rmagine_VERSION}
    COMPONENT core
)

# Uninstall
# https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rmagine-uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rmagine-uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/rmagine-uninstall.cmake
    COMMENT "Uninstall rmagine libraries and all header files")
endif()



###############################################################################
# DOCUMENTATION
###############################################################################

if(BUILD_DOCS)

find_package(Doxygen REQUIRED)

message(STATUS "Build Documentation with 'make doc'")

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

add_custom_target( doc ALL 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM )

endif(BUILD_DOCS)


#########################
# PACKAGE GENERATION
#########################
include(cmake/rmagine-deb.cmake)
include(CPack)
