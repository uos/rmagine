cmake_minimum_required(VERSION 3.4...10.0)
project(rmagine 
    LANGUAGES CXX C
    VERSION 2.2.1)# TODO update this version when merging into main-branch

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
# include_directories(
#     include
# )

add_definitions(${rmagine_DEFINITIONS})

set(RMAGINE_LIBRARIES)
set(RMAGINE_STATIC_LIBRARIES)

### RMAGINE CORE LIB
add_subdirectory(src/rmagine_core)

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

message(STATUS "RMAGINE_LIBRARIES: ${RMAGINE_LIBRARIES}")

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

configure_package_config_file(cmake/rmagineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

## INSTALL
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/rmagineConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
    COMPONENT core
)


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


#########################
# PACKAGE GENERATION
#########################
include(cmake/rmagine-deb.cmake)
include(CPack)










