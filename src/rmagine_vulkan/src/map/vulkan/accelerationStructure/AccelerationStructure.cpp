#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType) : 
    accelerationStructureType(accelerationStructureType),
    vulkan_context(get_vulkan_context_weak()),
    commandBuffer(new CommandBuffer(vulkan_context))
{
    
}

AccelerationStructure::~AccelerationStructure()
{
    std::cout << "Destroying AccelerationStructure" << std::endl;
    if(accelerationStructure != VK_NULL_HANDLE)
    {
        commandBuffer.reset();

        accelerationStructureDeviceMemory.reset();
        accelerationStructureBuffer.reset();
        accelerationStructureScratchDeviceMemory.reset();
        accelerationStructureScratchBuffer.reset();

        vulkan_context->extensionFuncs.vkDestroyAccelerationStructureKHR(vulkan_context->getDevice()->getLogicalDevice(), accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
    std::cout << "AccelerationStructure destroyed" << std::endl;
}

void AccelerationStructure::createAccelerationStructure(
    std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, 
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos)
{
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = accelerationStructureType;
    accelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = accelerationStructureGeometrys.size();
    accelerationStructureBuildGeometryInfo.pGeometries = accelerationStructureGeometrys.data();
    accelerationStructureBuildGeometryInfo.scratchData = {};
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = 0;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
    accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    accelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
    accelerationStructureBuildSizesInfo.updateScratchSize = 0;
    accelerationStructureBuildSizesInfo.buildScratchSize = 0;

    std::vector<uint32_t> maxPrimitiveCountList;
    for(size_t i = 0; i < accelerationStructureBuildRangeInfos.size(); i++)
    {
        maxPrimitiveCountList.push_back(accelerationStructureBuildRangeInfos[i].primitiveCount);
    }

    vulkan_context->extensionFuncs.vkGetAccelerationStructureBuildSizesKHR(
        vulkan_context->getDevice()->getLogicalDevice(), 
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, 
        &accelerationStructureBuildGeometryInfo, 
        maxPrimitiveCountList.data(), 
        &accelerationStructureBuildSizesInfo);

    //TODO: Memory Objects
    accelerationStructureBuffer = std::make_shared<Buffer>(
        accelerationStructureBuildSizesInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    accelerationStructureDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelerationStructureBuffer);

    VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
    accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreateInfo.buffer = accelerationStructureBuffer->getBuffer();
    accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
    accelerationStructureCreateInfo.type = accelerationStructureType;
    accelerationStructureCreateInfo.deviceAddress = 0;

    if(vulkan_context->extensionFuncs.vkCreateAccelerationStructureKHR(vulkan_context->getDevice()->getLogicalDevice(), &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to creates acceleration structure!");
    }
    
    VkAccelerationStructureDeviceAddressInfoKHR accelerationStructureDeviceAddressInfo{};
    accelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationStructureDeviceAddressInfo.accelerationStructure = accelerationStructure;

    accelerationStructureDeviceAddress = vulkan_context->extensionFuncs.vkGetAccelerationStructureDeviceAddressKHR(
        vulkan_context->getDevice()->getLogicalDevice(), 
        &accelerationStructureDeviceAddressInfo);
    
    //TODO: Memory Objects
    accelerationStructureScratchBuffer = std::make_shared<Buffer>(
        accelerationStructureBuildSizesInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    accelerationStructureScratchDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelerationStructureScratchBuffer);

    accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = accelerationStructureScratchBuffer->getBufferDeviceAddress();

    commandBuffer->recordBuildingASToCommandBuffer(accelerationStructureBuildGeometryInfo, accelerationStructureBuildRangeInfos.data());
    commandBuffer->submitRecordedCommandAndWait();
}


VkDeviceAddress AccelerationStructure::getDeviceAddress()
{
    return accelerationStructureDeviceAddress;
}


VkAccelerationStructureKHR* AccelerationStructure::getAcceleratiionStructurePtr()
{
    return &accelerationStructure;
}

} // namespace rmagine
