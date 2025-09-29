#include "rmagine/util/vulkan/ShaderBindingTable.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

ShaderBindingTable::ShaderBindingTable(DeviceWPtr device, RayTracingPipelineLayoutWPtr pipelineLayout, ShaderDefineFlags shaderDefines) : 
    device(device),
    pipeline(new RayTracingPipeline(device, pipelineLayout, shaderDefines)),
    shaderBindingTableMemory(0, VulkanMemoryUsage::Usage_ShaderBindingTable)
{
    std::cout << "[RMagine - ShaderBindingTable] creating sbt & pipeline: " << get_shader_defines_info(shaderDefines) << std::endl;
    createShaderBindingTable();
}


ShaderBindingTable::~ShaderBindingTable()
{
    if(pipeline != nullptr)
        pipeline.reset();
}



void ShaderBindingTable::createShaderBindingTable()
{
    VkDeviceSize progSize = device.lock()->getShaderGroupBaseAlignment();
    VkDeviceSize handleSize = device.lock()->getShaderGroupHandleSize();

    VkDeviceSize shaderBindingTableSize = progSize * 3;

    shaderBindingTableMemory.resize(shaderBindingTableSize);

    //get the group handels
    Memory<char, RAM> shaderBindingTableMemory_ram(shaderBindingTableSize);
    if(get_vulkan_context()->extensionFuncs.vkGetRayTracingShaderGroupHandlesKHR(device.lock()->getLogicalDevice(), pipeline->getPipeline(), 0, 3, shaderBindingTableSize, shaderBindingTableMemory_ram.raw()) != VK_SUCCESS)
    {
        throw std::runtime_error("[ShaderBindingTable::createShaderBindingTable()] ERROR - Failed to get raytracing shadergroup handles!");
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


RayTracingPipelinePtr ShaderBindingTable::getPipeline()
{
    return pipeline;
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
