#include "rmagine/util/vulkan/ShaderBindingTable.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

ShaderBindingTable::ShaderBindingTable(PipelinePtr pipeline) : device(get_vulkan_context()->getDevice()), extensionFunctionsPtr(get_vulkan_context()->getExtensionFunctionsPtr())
{
    createShaderBindingTable(pipeline);
}


ShaderBindingTable::ShaderBindingTable(DevicePtr device, PipelinePtr pipeline, ExtensionFunctionsPtr extensionFunctionsPtr) : device(device), extensionFunctionsPtr(extensionFunctionsPtr)
{
    createShaderBindingTable(pipeline);
}



void ShaderBindingTable::createShaderBindingTable(PipelinePtr pipeline)
{
    VkDeviceSize progSize = device->getShaderGroupBaseAlignment();
    VkDeviceSize handleSize = device->getShaderGroupHandleSize();

    VkDeviceSize shaderBindingTableSize = progSize * 3;

    shaderBindingTableMemory.resize(shaderBindingTableSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR);

    //get the group handels
    Memory<char, RAM> shaderBindingTableMemory_ram(shaderBindingTableSize);
    if(extensionFunctionsPtr->pvkGetRayTracingShaderGroupHandlesKHR(device->getLogicalDevice(), pipeline->getPipeline(), 0, 3, shaderBindingTableSize, shaderBindingTableMemory_ram.raw()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to get raytracing shadergroup handles!");
    }

    //only a part of the data is needed
    Memory<char, RAM> shaderBindingTableMemory_ram_2(shaderBindingTableSize);
    for(size_t i = 0; i < 3; i++)
    {
        memcpy(shaderBindingTableMemory_ram_2.raw() + i * progSize, shaderBindingTableMemory_ram.raw() + i * handleSize, handleSize);
    }

    //load group handels onto gpu
    shaderBindingTableMemory = shaderBindingTableMemory_ram_2;


    VkDeviceSize hitGroupOffset = 0u * progSize;
    VkDeviceSize rayGenOffset = 1u * progSize;
    VkDeviceSize missOffset = 2u * progSize;

    rchitShaderBindingTable.deviceAddress = shaderBindingTableMemory.getBuffer()->getBufferDeviceAddress() + hitGroupOffset;
    rchitShaderBindingTable.stride = progSize;
    rchitShaderBindingTable.size = progSize;

    rgenShaderBindingTable.deviceAddress = shaderBindingTableMemory.getBuffer()->getBufferDeviceAddress() + rayGenOffset;
    rgenShaderBindingTable.stride = progSize;
    rgenShaderBindingTable.size = progSize;

    rmissShaderBindingTable.deviceAddress = shaderBindingTableMemory.getBuffer()->getBufferDeviceAddress() + missOffset;
    rmissShaderBindingTable.stride = progSize;
    rmissShaderBindingTable.size = progSize;

    callableShaderBindingTable.deviceAddress = 0;
    callableShaderBindingTable.stride = 0;
    callableShaderBindingTable.size = 0;
}


void ShaderBindingTable::cleanup()
{
    //...
}



VkStridedDeviceAddressRegionKHR* ShaderBindingTable::getRayGenerationShaderBindingTablePtr()
{
    return &rgenShaderBindingTable;
}

VkStridedDeviceAddressRegionKHR* ShaderBindingTable::getClosestHitShaderBindingTablePtr()
{
    return &rchitShaderBindingTable;
}

VkStridedDeviceAddressRegionKHR* ShaderBindingTable::getMissShaderBindingTablePtr()
{
    return &rmissShaderBindingTable;
}

VkStridedDeviceAddressRegionKHR* ShaderBindingTable::getCallableShaderBindingTablePtr()
{
    return &callableShaderBindingTable;
}

} // namespace rmagine
