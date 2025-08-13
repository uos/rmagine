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
    PFN_vkGetBufferDeviceAddressKHR pvkGetBufferDeviceAddressKHR;
    PFN_vkCreateRayTracingPipelinesKHR pvkCreateRayTracingPipelinesKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR pvkGetAccelerationStructureBuildSizesKHR;
    PFN_vkCreateAccelerationStructureKHR pvkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR pvkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pvkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR pvkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR pvkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR pvkCmdTraceRaysKHR;
};

using ExtensionFunctionsPtr = std::shared_ptr<ExtensionFunctions>;

} // namespace rmagine
