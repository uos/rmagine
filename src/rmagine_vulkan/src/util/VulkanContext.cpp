#include "VulkanContext.hpp"
#include "contextComponents/ShaderBindingTable.hpp"



namespace rmagine
{

void VulkanContext::loadExtensionFunctions()
{
    extensionFunctionsPtr->pvkGetBufferDeviceAddressKHR =
        (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetBufferDeviceAddressKHR");

    extensionFunctionsPtr->pvkCreateRayTracingPipelinesKHR =
        (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCreateRayTracingPipelinesKHR");

    extensionFunctionsPtr->pvkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetAccelerationStructureBuildSizesKHR");

    extensionFunctionsPtr->pvkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCreateAccelerationStructureKHR");

    extensionFunctionsPtr->pvkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkDestroyAccelerationStructureKHR");

    extensionFunctionsPtr->pvkGetAccelerationStructureDeviceAddressKHR =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetAccelerationStructureDeviceAddressKHR");

    extensionFunctionsPtr->pvkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCmdBuildAccelerationStructuresKHR");

    extensionFunctionsPtr->pvkGetRayTracingShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetRayTracingShaderGroupHandlesKHR");

    extensionFunctionsPtr->pvkCmdTraceRaysKHR =
        (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCmdTraceRaysKHR");
}


PipelinePtr VulkanContext::getPipeline(ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::invalid_argument("illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::END)
    {
        throw std::invalid_argument("illegal ShaderDefineFlags: cant be 0 or too large");
    }

    if(pipelineMap.count(shaderDefines) == 0)
    {
        pipelineMap[shaderDefines] = std::make_shared<Pipeline>(device, pipelineLayout, extensionFunctionsPtr, shaderDefines);
        pipelineMap.at(shaderDefines)->createShaderBindingTable();
    }

    return pipelineMap.at(shaderDefines);
}


ShaderPtr VulkanContext::getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::runtime_error("illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::END)
    {
        throw std::invalid_argument("illegal ShaderDefineFlags: cant be 0 or too large!");
    }

    ShaderDefineFlags maskedShaderDefines = 0;
    switch (shaderType)
    {
    case ShaderType::RGen:
        maskedShaderDefines = shaderDefines & get_sensor_mask();
        break;
    
    case ShaderType::Miss:
    case ShaderType::CHit:
        maskedShaderDefines = shaderDefines & get_result_mask();
        break;
    
    case ShaderType::Call:
        throw std::runtime_error("ShaderType::Call currently not supported!");
        break;
    default:
        throw std::invalid_argument("invalid shaderType");
        break;
    }

    if(shaderMaps[shaderType].count(maskedShaderDefines) == 0)
    {
        shaderMaps[shaderType][maskedShaderDefines] = std::make_shared<Shader>(device, shaderType, maskedShaderDefines);
    }

    return shaderMaps[shaderType].at(maskedShaderDefines);
}


void VulkanContext::cleanup()
{
    std::cout << "cleaning up..." << std::endl;

    clearShaderCache();
    std::cout << "cleaned up shaders." << std::endl;

    defaultCommandBuffer->cleanup();
    commandPool->cleanup();
    std::cout << "reset & cleaned up command pool." << std::endl;

    clearPipelineCache();
    std::cout << "cleaned up pipelines." << std::endl;

    pipelineLayout->cleanup();
    std::cout << "cleaned up pipeline layout." << std::endl;

    descriptorSetLayout->cleanup();
    std::cout << "cleaned up descriptor set layout." << std::endl;

    device->cleanup();
    std::cout << "cleaned up device & instance." << std::endl;

    std::cout << "done." << std::endl;
}



void VulkanContext::clearShaderCache()
{
    for(size_t i = 0; i < ShaderType::SIZE; i++)
    {
        for (auto const& shader : shaderMaps[i])
        {
            shader.second->cleanup();
        }
        shaderMaps[i].clear();
    }
}

void VulkanContext::clearPipelineCache()
{
    for (auto const& pipeline : pipelineMap)
    {
        pipeline.second->cleanup();
    }
    pipelineMap.clear();
}



DevicePtr VulkanContext::getDevice()
{
    return device;
}

CommandPoolPtr VulkanContext::getCommandPool()
{
    return commandPool;
}

DescriptorSetLayoutPtr VulkanContext::getDescriptorSetLayout()
{
    return descriptorSetLayout;
}

PipelineLayoutPtr VulkanContext::getPipelineLayout()
{
    return pipelineLayout;
}

ExtensionFunctionsPtr VulkanContext::getExtensionFunctionsPtr()
{
    return extensionFunctionsPtr;
}

CommandBufferPtr VulkanContext::getDefaultCommandBuffer()
{
    return defaultCommandBuffer;
}

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------End of VulkanContext functions-------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

VulkanContextPtr vulkan_context(new VulkanContext());

VulkanContextPtr get_vulkan_context()
{
    return vulkan_context;
}

VulkanContextWeakPtr get_vulkan_context_weak()
{
    return vulkan_context;
}

} // namespace rmagine
