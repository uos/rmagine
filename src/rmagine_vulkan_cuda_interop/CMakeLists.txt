####################################
## Package: rmagine               ##
## Component: vulkan_cuda_interop ##
####################################

message(STATUS "Building Vulkan Cuda Interop. Library: rmagine-vulkan-cuda-interop")

set(RMAGINE_VULKAN_CUDA_INTEROP_SRCS
    #Types
    src/types/VulkanCudaInterop.cpp
)

## SHARED ##
add_library(rmagine-vulkan-cuda-interop SHARED
    ${RMAGINE_VULKAN_CUDA_INTEROP_SRCS}
)

target_include_directories(rmagine-vulkan-cuda-interop
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
        # ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(rmagine-vulkan-cuda-interop
    rmagine-core
    rmagine-vulkan
    rmagine-cuda
)

add_dependencies(rmagine-vulkan-cuda-interop
    rmagine-core
    rmagine-vulkan
    rmagine-cuda
)

target_compile_definitions(rmagine-vulkan-cuda-interop PUBLIC VDEBUG) #TODO: remove later

# target_compile_features(rmagine-vulkan-cuda-interop PUBLIC cxx_std_17)

set_target_properties(rmagine-vulkan-cuda-interop
    PROPERTIES
        EXPORT_NAME vulkanCudaInterop
        SOVERSION ${rmagine_VERSION_MAJOR}
        VERSION ${rmagine_VERSION}
        # CXX_STANDARD 17
)

add_library(rmagine::vulkanCudaInterop ALIAS rmagine-vulkan-cuda-interop)

list(APPEND RMAGINE_LIBRARIES rmagine-vulkan-cuda-interop)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

###########
## INSTALL
############

include(CMakePackageConfigHelpers)

install(TARGETS rmagine-vulkan-cuda-interop EXPORT rmagine-vulkan-cuda-interop-targets
    COMPONENT vulkanCudaInterop
)

install(EXPORT rmagine-vulkan-cuda-interop-targets
    FILE rmagine-vulkan-cuda-interop-targets.cmake
    COMPONENT vulkanCudaInterop
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config-version.cmake
    VERSION ${rmagine_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-vulkan-cuda-interop-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

# CMAKE FIND PACKAGE FILES
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT vulkanCudaInterop
)

# HEADERS
install(
    DIRECTORY include/rmagine
    COMPONENT vulkanCudaInterop
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)
# set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-vulkan" PARENT_SCOPE)
# set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-cuda" PARENT_SCOPE)
# TODO: was davon wäre richtig?

list(APPEND CPACK_COMPONENTS_ALL vulkanCudaInterop)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)
