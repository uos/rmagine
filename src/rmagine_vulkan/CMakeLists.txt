##########################
## Package: rmagine     ##
## Component: vulkan    ##
##########################

message(STATUS "Building Vulkan backend. Library: rmagine-vulkan")

set(RMAGINE_VULKAN_SRCS
    # Util
    src/util/VulkanContext.cpp
    #   Context components
    src/util/vulkan/Device.cpp
    src/util/vulkan/CommandPool.cpp
    src/util/vulkan/DescriptorSetLayout.cpp
    src/util/vulkan/PipelineLayout.cpp
    src/util/vulkan/Pipeline.cpp
    src/util/vulkan/ShaderBindingTable.cpp
    src/util/vulkan/Shader.cpp
    src/util/vulkan/ShaderUtil.cpp
    #   Command components
    src/util/vulkan/command/Fence.cpp
    src/util/vulkan/command/CommandBuffer.cpp
    #   Memory components
    src/util/vulkan/memory/Buffer.cpp
    src/util/vulkan/memory/DeviceMemory.cpp

    #Types
    src/types/MemoryVulkan.cpp
    src/types/MemoryVulkanUtil.cpp

    # Maps
    src/map/VulkanMap.cpp
    #   Map Components
    src/map/vulkan/VulkanScene.cpp
    src/map/vulkan/VulkanMesh.cpp
    src/map/vulkan/VulkanInst.cpp
    src/map/vulkan/VulkanGeometry.cpp
    src/map/vulkan/VulkanTransformable.cpp
    #   Acceleration structure
    src/map/vulkan/accelerationStructure/AccelerationStructure.cpp
    src/map/vulkan/accelerationStructure/TopLevelAccelerationStructure.cpp
    src/map/vulkan/accelerationStructure/BottomLevelAccelerationStructure.cpp

    # Simulators
    src/simulation/SphereSimulatorVulkan.cpp
    src/simulation/PinholeSimulatorVulkan.cpp
    src/simulation/O1DnSimulatorVulkan.cpp
    src/simulation/OnDnSimulatorVulkan.cpp
    #   Simulator components
    src/simulation/vulkan/DescriptorSet.cpp
)

## SHARED ##
add_library(rmagine-vulkan SHARED
    ${RMAGINE_VULKAN_SRCS}
)

target_include_directories(rmagine-vulkan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
        # ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(rmagine-vulkan
    rmagine-core
    ${Vulkan_LIBRARIES}
    glslang::glslang 
    glslang::SPIRV 
    glslang::glslang-default-resource-limits
)

add_dependencies(rmagine-vulkan
    rmagine-core
)

target_compile_definitions(rmagine-vulkan PUBLIC VDEBUG) #TODO: remove later

# target_compile_features(rmagine-vulkan PUBLIC cxx_std_17)

set_target_properties(rmagine-vulkan
    PROPERTIES
        EXPORT_NAME vulkan
        SOVERSION ${rmagine_VERSION_MAJOR}
        VERSION ${rmagine_VERSION}
        # CXX_STANDARD 17
)

add_library(rmagine::vulkan ALIAS rmagine-vulkan)

list(APPEND RMAGINE_LIBRARIES rmagine-vulkan)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

###########
## INSTALL
############

include(CMakePackageConfigHelpers)

install(TARGETS rmagine-vulkan EXPORT rmagine-vulkan-targets
    COMPONENT vulkan
)

install(EXPORT rmagine-vulkan-targets
    FILE rmagine-vulkan-targets.cmake
    COMPONENT vulkan
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-config-version.cmake
    VERSION ${rmagine_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-vulkan-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

# CMAKE FIND PACKAGE FILES
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT vulkan
)

# HEADERS
install(
    DIRECTORY include/rmagine
    COMPONENT vulkan
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_VULKAN_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL vulkan)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)
