cmake_minimum_required(VERSION 3.16)
project(rmagine 
    LANGUAGES CXX C
    VERSION 2.3.0)# TODO update this version when merging into main-branch

option(RMAGINE_BUILD_TOOLS "Build tools" ON)
option(RMAGINE_BUILD_TESTS "Build tests" ON)
option(RMAGINE_BUILD_EXAMPLES "Build examples" OFF)
option(RMAGINE_BUILD_DOCS "Build documentation" OFF)

set(RMAGINE_CPM_MODE "Auto" CACHE STRING "Choose the dependency management mode for CPM.")
set_property(CACHE RMAGINE_CPM_MODE PROPERTY STRINGS "Local" "Auto" "Download" "None")
# Local: Use local packages only. This is the old default behavior until v2.3.0
# Auto: Use local package if available and download otherwise (default from v2.3.1)
# Download: Always download.
# None: Let CPM decide. You can setup the behavior by setting the respective CPM variables manually

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

list(APPEND CMAKE_MODULE_PATH ${rmagine_SOURCE_DIR}/cmake)

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


include(cmake/CPM.cmake)

################################
## REQUIRED DEPS

find_package(Boost CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

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

######################
## Build Components ##
######################

add_definitions(${rmagine_DEFINITIONS})

set(RMAGINE_LIBRARIES)
set(RMAGINE_STATIC_LIBRARIES)

### RMAGINE CORE COMPONENT
add_subdirectory(src/rmagine_core)

### RMAGINE OUSTER COMPONENT
option(RMAGINE_OUSTER_DISABLE "Disable rmagine::ouster compilation" OFF)
if(NOT RMAGINE_OUSTER_DISABLE)
    add_subdirectory(src/rmagine_ouster)
endif()

### RMAGINE EMBREE COMPONENT
option(RMAGINE_EMBREE_DISABLE "Disable rmagine::embree compilation" OFF)
if(NOT RMAGINE_EMBREE_DISABLE)
    add_subdirectory(src/rmagine_embree)
endif()

### RMAGINE CUDA COMPONENT
option(RMAGINE_CUDA_DISABLE "Disable rmagine::cuda compilation" OFF)
if(NOT RMAGINE_CUDA_DISABLE)
    add_subdirectory(src/rmagine_cuda)
endif(NOT RMAGINE_CUDA_DISABLE)

### RMAGINE OPTIX COMPONENT
option(RMAGINE_OPTIX_DISABLE "Disable rmagine::optix compilation" OFF)
if(NOT RMAGINE_OPTIX_DISABLE)
    add_subdirectory(src/rmagine_optix)
endif(NOT RMAGINE_OPTIX_DISABLE)

message(STATUS "${BoldCyan}Components being built:${ColourReset}")
foreach(LIBRARY ${RMAGINE_LIBRARIES})
    message(STATUS "- ${BoldGreen}${LIBRARY}${ColourReset}")
endforeach()

#### TESTS
if(RMAGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(RMAGINE_BUILD_TESTS)

#### TOOLS
if(RMAGINE_BUILD_TOOLS)
    add_subdirectory(apps/rmagine_benchmark)
    add_subdirectory(apps/rmagine_synthetic)
    add_subdirectory(apps/rmagine_map_info)
    add_subdirectory(apps/rmagine_version)
    add_subdirectory(apps/rmagine_info)
endif(RMAGINE_BUILD_TOOLS)

if(RMAGINE_BUILD_EXAMPLES)
if(TARGET rmagine-embree)
    add_subdirectory(apps/rmagine_examples/rmagine_embree_scene)
endif(TARGET rmagine-embree)

if(TARGET rmagine-optix)
    add_subdirectory(apps/rmagine_examples/rmagine_optix_scene)
endif(TARGET rmagine-optix)
endif(RMAGINE_BUILD_EXAMPLES)


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

if(RMAGINE_BUILD_DOCS)

find_package(Doxygen REQUIRED)

message(STATUS "Build Documentation with 'make doc'")

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )

add_custom_target( doc ALL 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM )

endif(RMAGINE_BUILD_DOCS)


#########################
# PACKAGE GENERATION
#########################
include(cmake/rmagine-deb.cmake)
include(CPack)
