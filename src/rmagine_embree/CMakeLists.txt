##########################
## Package: rmagine     ##
## Component: embree    ##
##########################

##################################
## Required external libraries: ##
## - embree                     ##
##################################

# version of embree that is used for searching a local package
set(RMAGINE_EMBREE_LOCAL_VERSION 4)
# version of embree that is used for download
set(RMAGINE_EMBREE_DOWNLOAD_VERSION "4.4.0" CACHE STRING "Desired embree version used for download.")

message(STATUS "Checking dependencies of rmagine::embree ...")


set(TMP_LIST_AUTO_LOCAL "Auto;Local")
if(${RMAGINE_DEPENDENCY_MODE} IN_LIST TMP_LIST_AUTO_LOCAL)
  find_package(embree ${RMAGINE_EMBREE_LOCAL_VERSION} QUIET)
endif()

############
# DOWNLOAD EMBREE
# Only when embree is not available and RMAGINE_DEPENDENCY_MODE is set to either "Auto" or "Download"
# -> In case of download all rmagine targets get extra interface links to vendor libraries 
# that are installed to custom locations. If the locations are empty you can still install Embree
# the standard way and ld finds the missing libraries again 
set(TMP_LIST_AUTO_DOWNLOAD "Auto;Download")
if(NOT embree_FOUND AND ${RMAGINE_DEPENDENCY_MODE} IN_LIST TMP_LIST_AUTO_DOWNLOAD)

    message(STATUS "Search for Embree online ...")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
      # 64 Bit linux shortcut
      message(STATUS "Trying to download precompiled Embree. Linux ${CMAKE_SYSTEM_PROCESSOR}")

      FetchContent_Declare(
        embree_bin
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL https://github.com/RenderKit/embree/releases/download/v${RMAGINE_EMBREE_DOWNLOAD_VERSION}/embree-${RMAGINE_EMBREE_DOWNLOAD_VERSION}.x86_64.linux.tar.gz
      )
      FetchContent_Populate(embree_bin)

      set(EMBREE_ROOT "${embree_bin_SOURCE_DIR}")

      
      # Imported target for Embree 4 (adjust names if your archive differs)
      add_library(embree SHARED IMPORTED GLOBAL)

      set_target_properties(embree PROPERTIES
        IMPORTED_LOCATION "${EMBREE_ROOT}/lib/libembree4.so"         # Linux
        INTERFACE_INCLUDE_DIRECTORIES "${EMBREE_ROOT}/include"
      )

      # Install the vendored payload with your project
      install(DIRECTORY "${EMBREE_ROOT}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
      install(DIRECTORY "${EMBREE_ROOT}/lib/"     DESTINATION ${CMAKE_INSTALL_LIBDIR})

      set(embree_FOUND TRUE)

    else()
      message(STATUS "No precompiled binaries found for ${CMAKE_SYSTEM_NAME}:${CMAKE_SYSTEM_PROCESSOR}")

      # Else: Download sources and compile
      message(STATUS "-> Compile Embree from source. This might take a while.")

      # 1) Declare the source
      FetchContent_Declare(
        embree
        GIT_REPOSITORY https://github.com/RenderKit/embree
        GIT_TAG v${RMAGINE_EMBREE_DOWNLOAD_VERSION}
      )

      # 2) Set Embree's options BEFORE it's added
      set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "" FORCE)
      set(EMBREE_TUTORIALS    OFF CACHE BOOL "" FORCE)
      set(EMBREE_TESTING      OFF CACHE BOOL "" FORCE)
      set(EMBREE_TUTORIALS_GLFW OFF CACHE BOOL "" FORCE)

      # 3) Add the dependency (this does add_subdirectory on Embree)
      FetchContent_MakeAvailable(embree)

      message(STATUS "INCLUDE: ${EMBREE_INCLUDE_DIR}")

      set(embree_FOUND TRUE)
    endif()
endif()

message(STATUS "EMBREE_LIBRARY: ${EMBREE_LIBRARY}")

if(embree_FOUND)

message(STATUS "${Green}rmagine::embree${ColourReset}: Building (Embree v${EMBREE_VERSION})")

set(RMAGINE_EMBREE_SRCS
  # Maps
  src/map/embree/EmbreeDevice.cpp
  src/map/embree/EmbreeGeometry.cpp
  src/map/embree/EmbreeMesh.cpp
  src/map/embree/EmbreeScene.cpp
  src/map/embree/EmbreeInstance.cpp
  src/map/embree/EmbreePoints.cpp
  src/map/embree/embree_shapes.cpp
  src/map/EmbreeMap.cpp

  # Simulators
  src/simulation/SimulatorEmbree.cpp
  src/simulation/SphereSimulatorEmbree.cpp
  src/simulation/PinholeSimulatorEmbree.cpp
  src/simulation/O1DnSimulatorEmbree.cpp
  src/simulation/OnDnSimulatorEmbree.cpp
)

## SHARED ##
add_library(rmagine-embree SHARED
  ${RMAGINE_EMBREE_SRCS}
)

target_include_directories(rmagine-embree
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
)

target_link_libraries(rmagine-embree PUBLIC
  rmagine-core
  embree
)

# target_compile_features(rmagine-embree PUBLIC cxx_std_17)
set_target_properties(rmagine-embree
    PROPERTIES
      EXPORT_NAME embree
      SOVERSION ${rmagine_VERSION_MAJOR}
      VERSION ${rmagine_VERSION}
      # CXX_STANDARD 17
)

add_library(rmagine::embree ALIAS rmagine-embree)

list(APPEND RMAGINE_LIBRARIES rmagine-embree)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

###########
## INSTALL
############

include(CMakePackageConfigHelpers)

install(TARGETS rmagine-embree EXPORT rmagine-embree-targets
  COMPONENT embree
)

install(EXPORT rmagine-embree-targets
    FILE rmagine-embree-targets.cmake
    COMPONENT embree
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config-version.cmake
  VERSION ${rmagine_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-embree-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

# CMAKE FIND PACKAGE FILES
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-embree-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
  COMPONENT embree
)

# HEADERS
install(
  DIRECTORY include/rmagine
  COMPONENT embree
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_EMBREE_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL embree)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)

else(embree_FOUND)

message(STATUS "${Yellow}rmagine::embree${ColourReset}: Skipping. Missing Dependencies.")

endif(embree_FOUND)
