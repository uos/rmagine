#include "rmagine/util/VulkanContext.hpp"
#include "rmagine/util/vulkan/ShaderBindingTable.hpp"



namespace rmagine
{

VulkanContext::VulkanContext() : device(new Device)
{
    std::cout << "Creating VulkanContext" << std::endl;

    // device = std::make_shared<Device>();
    loadExtensionFunctions();

    std::cout << "VulkanContext created" << std::endl;
}

VulkanContext::~VulkanContext()
{
    std::cout << "Destroying VulkanContext" << std::endl;
    clearShaderBindingTableCache();
    std::cout << "Destroying VulkanContext2" << std::endl;
    clearShaderCache();
    std::cout << "Destroying VulkanContext3" << std::endl;

    commandPool.reset();
    std::cout << "Destroying VulkanContext4" << std::endl;
    pipelineLayout.reset();
    std::cout << "Destroying VulkanContext5" << std::endl;
    descriptorSetLayout.reset();
    std::cout << "Destroying VulkanContext6" << std::endl;

    device.reset();
    std::cout << "VulkanContext destroyed" << std::endl;
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
}


ShaderBindingTablePtr VulkanContext::getShaderBindingTable(ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::invalid_argument("illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
    {
        throw std::invalid_argument("illegal ShaderDefineFlags: cant be 0 or too large");
    }

    if(shaderBindingTableMap.count(shaderDefines) == 0)
    {
        shaderBindingTableMap[shaderDefines] = std::make_shared<ShaderBindingTable>(weak_from_this(), shaderDefines);
    }

    return shaderBindingTableMap.at(shaderDefines);
}


ShaderPtr VulkanContext::getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    if(!one_sensor_defined(shaderDefines))
    {
        throw std::runtime_error("illegal ShaderDefineFlags: You may only define one sensor type!");
    }
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
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
        shaderMaps[shaderType][maskedShaderDefines] = std::make_shared<Shader>(weak_from_this(), shaderType, maskedShaderDefines);
    }

    return shaderMaps[shaderType].at(maskedShaderDefines);
}


void VulkanContext::clearShaderCache()
{
    for(size_t i = 0; i < ShaderType::SHADER_TYPE_SIZE; i++)
    {
        shaderMaps[i].clear();
    }
}


void VulkanContext::clearShaderBindingTableCache()
{
    shaderBindingTableMap.clear();
}


DevicePtr VulkanContext::getDevice()
{
    return device;
}

CommandPoolPtr VulkanContext::getCommandPool()
{
    if(commandPool == VK_NULL_HANDLE)
    {
        commandPool = std::make_shared<CommandPool>(weak_from_this());
    }
    return commandPool;
}

DescriptorSetLayoutPtr VulkanContext::getDescriptorSetLayout()
{
    if(descriptorSetLayout == VK_NULL_HANDLE)
    {
        descriptorSetLayout = std::make_shared<DescriptorSetLayout>(weak_from_this());
    }
    return descriptorSetLayout;
}

PipelineLayoutPtr VulkanContext::getPipelineLayout()
{
    if(pipelineLayout == VK_NULL_HANDLE)
    {
        pipelineLayout = std::make_shared<PipelineLayout>(weak_from_this());
    }
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
