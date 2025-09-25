#include "rmagine/util/VulkanContext.hpp"
#include "rmagine/util/vulkan/ShaderBindingTable.hpp"



namespace rmagine
{

VulkanContext::VulkanContext() : 
    device(new Device), commandPool(new CommandPool(device)), 
    descriptorSetLayout(new DescriptorSetLayout(device)), pipelineLayout(new RayTracingPipelineLayout(device, descriptorSetLayout))
{
    loadExtensionFunctions();
}

VulkanContext::~VulkanContext()
{
    clearShaderBindingTableCache();
    clearShaderCache();

    commandPool.reset();
    pipelineLayout.reset();
    descriptorSetLayout.reset();
    device.reset();
}



void VulkanContext::loadExtensionFunctions()
{
    extensionFuncs.vkCreateRayTracingPipelinesKHR =
        (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCreateRayTracingPipelinesKHR");

    extensionFuncs.vkGetAccelerationStructureBuildSizesKHR =
        (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetAccelerationStructureBuildSizesKHR");

    extensionFuncs.vkCreateAccelerationStructureKHR =
        (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCreateAccelerationStructureKHR");

    extensionFuncs.vkDestroyAccelerationStructureKHR =
        (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkDestroyAccelerationStructureKHR");

    extensionFuncs.vkGetAccelerationStructureDeviceAddressKHR =
        (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetAccelerationStructureDeviceAddressKHR");

    extensionFuncs.vkCmdBuildAccelerationStructuresKHR =
        (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCmdBuildAccelerationStructuresKHR");

    extensionFuncs.vkGetRayTracingShaderGroupHandlesKHR =
        (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetRayTracingShaderGroupHandlesKHR");

    extensionFuncs.vkCmdTraceRaysKHR =
        (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkCmdTraceRaysKHR");

    extensionFuncs.vkGetMemoryFdKHR = 
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device->getLogicalDevice(), "vkGetMemoryFdKHR");

}


ShaderPtr VulkanContext::getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::invalid_argument("[VulkanContext::getShader()] ERROR - illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
    {
        throw std::invalid_argument("[VulkanContext::getShader()] ERROR - illegal ShaderDefineFlags: cant be 0 or too large!");
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
        throw std::invalid_argument("[VulkanContext::getShader()] ERROR - ShaderType::Call currently not supported!");
        break;
    default:
        throw std::invalid_argument("[VulkanContext::getShader()] ERROR - invalid shaderType");
        break;
    }

    std::lock_guard<std::mutex> guard(shaderMutex);
    if(shaderMaps[shaderType].count(maskedShaderDefines) == 0)
    {
        shaderMaps[shaderType][maskedShaderDefines] = std::make_shared<Shader>(weak_from_this(), shaderType, maskedShaderDefines);
    }

    return shaderMaps[shaderType].at(maskedShaderDefines);
}


void VulkanContext::removeShader(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    std::lock_guard<std::mutex> guard(shaderMutex);
    if(shaderMaps[shaderType].count(shaderDefines) == 1)
    {
        auto it = shaderMaps[shaderType].find(shaderDefines);
        shaderMaps[shaderType].erase(it);
    }
}


size_t VulkanContext::getShaderCacheSize()
{
    std::lock_guard<std::mutex> guard(shaderMutex);
    size_t size = 0;
    for(size_t i = 0; i < ShaderType::SHADER_TYPE_SIZE; i++)
    {
        size += shaderMaps[i].size();
    }
    return size;
}


void VulkanContext::clearShaderCache()
{
    std::lock_guard<std::mutex> guard(shaderMutex);
    for(size_t i = 0; i < ShaderType::SHADER_TYPE_SIZE; i++)
    {
        shaderMaps[i].clear();
    }
}


ShaderBindingTablePtr VulkanContext::getShaderBindingTable(ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::invalid_argument("[VulkanContext::getShaderBindingTable()] ERROR - illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
    {
        throw std::invalid_argument("[VulkanContext::getShaderBindingTable()] ERROR - illegal ShaderDefineFlags: cant be 0 or too large");
    }

    std::lock_guard<std::mutex> guard(sbtMutex);
    if(shaderBindingTableMap.count(shaderDefines) == 0)
    {
        shaderBindingTableMap[shaderDefines] = std::make_shared<ShaderBindingTable>(weak_from_this(), shaderDefines);
    }

    return shaderBindingTableMap.at(shaderDefines);
}


void VulkanContext::removeShaderBindingTable(ShaderDefineFlags shaderDefines)
{
    std::lock_guard<std::mutex> guard(sbtMutex);
    if(shaderBindingTableMap.count(shaderDefines) == 1)
    {
        auto it = shaderBindingTableMap.find(shaderDefines);
        shaderBindingTableMap.erase(it);
    }
}


size_t VulkanContext::getShaderBindingTableCacheSize()
{
    std::lock_guard<std::mutex> guard(sbtMutex);
    return shaderBindingTableMap.size();
}


void VulkanContext::clearShaderBindingTableCache()
{
    std::lock_guard<std::mutex> guard(sbtMutex);
    shaderBindingTableMap.clear();
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

RayTracingPipelineLayoutPtr VulkanContext::getPipelineLayout()
{
    return pipelineLayout;
}

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------End of VulkanContext functions-------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

VulkanContextPtr vulkan_context = std::make_shared<VulkanContext>();

VulkanContextPtr get_vulkan_context()
{
    return vulkan_context;
}

VulkanContextWPtr get_vulkan_context_weak()
{
    return vulkan_context;
}

} // namespace rmagine
