#pragma once

#define _USE_MATH_DEFINES // needed on some devices for some reason
#include <math.h>
#include <memory>

#include <vulkan/vulkan.h>



namespace rmagine
{

/**
 * sets the maximum number of allowed descriptor sets.
 * thus this also describes how many simulators can be created,
 * as each simultor needs one descriptor set.
 * 
 * this number should be high enough to the user to create enough simulators,
 * but setting it too high leads to unnecceassarily allocating unneeded memory.
 * 
 * this is set at progam startup when crating the VkDescriptorPool in the DescriptorSetLayout Class.
 * this number cannot be updated later on, the program would need to create a new VkDescriptorPool.
 */
#define MAX_NUM_OF_DESCRIPTOR_SETS 256



struct ExtensionFunctions
{
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
};



//forward declaration
class VulkanContext;
using VulkanContextPtr = std::shared_ptr<VulkanContext>;
using VulkanContextWPtr = std::weak_ptr<VulkanContext>;

} // namespace rmagine
