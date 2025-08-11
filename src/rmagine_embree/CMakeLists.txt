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
set(RMAGINE_EMBREE_DOWNLOAD_VERSION 4.4.0)

message(STATUS "Checking dependencies of rmagine::embree ...")

set(TMP_LIST_AUTO_LOCAL "Auto;Local")
if(${RMAGINE_CPM_MODE} IN_LIST TMP_LIST_AUTO_LOCAL)
    find_package(embree ${RMAGINE_EMBREE_LOCAL_VERSION} QUIET)
endif()

set(TMP_LIST_AUTO_DOWNLOAD "Auto;Download")
if(NOT embree_FOUND AND ${RMAGINE_CPM_MODE} IN_LIST TMP_LIST_AUTO_DOWNLOAD)

    CPMAddPackage(
        URI "gh:RenderKit/embree"
        VERSION ${RMAGINE_EMBREE_DOWNLOAD_VERSION}
        OPTIONS
            "EMBREE_TUTORIALS OFF"
            "EMBREE_TESTING OFF"
            "EMBREE_ISPC_SUPPORT OFF"   # turn ON if you use ISPC and have the compiler
            "BUILD_SHARED_LIBS ON"
    )
    if(TARGET embree)
        set(embree_FOUND TRUE)
        set(embree_LIBRARY embree)
        set(embree_VERSION ${RMAGINE_EMBREE_DOWNLOAD_VERSION})
    else()
        set(embree_FOUND FALSE)
    endif(TARGET embree)
endif()


if(embree_FOUND)

message(STATUS "${Green}rmagine::embree${ColourReset}: Building (Embree v${embree_VERSION})")

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

target_link_libraries(rmagine-embree
    rmagine-core
    ${embree_LIBRARY}
)

add_dependencies(rmagine-embree
    rmagine-core
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
