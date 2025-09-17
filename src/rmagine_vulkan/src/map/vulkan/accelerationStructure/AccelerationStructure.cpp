#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType) : 
    accelerationStructureType(accelerationStructureType),
    vulkan_context(get_vulkan_context_weak()),
    commandBuffer(new CommandBuffer(vulkan_context)),
    accelerationStructureMem(0, VulkanMemoryUsage::Usage_AccelerationStructure),
    accelerationStructureScratchMem(0, VulkanMemoryUsage::Usage_AccelerationStructureScratch)
{
    
}

AccelerationStructure::~AccelerationStructure()
{
    std::cout << "Destroying AccelerationStructure" << std::endl;
    if(accelerationStructure != VK_NULL_HANDLE)
    {
        commandBuffer.reset();

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

    accelerationStructureMem.resize(accelerationStructureBuildSizesInfo.accelerationStructureSize);

    VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
    accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreateInfo.buffer = accelerationStructureMem.getBuffer()->getBuffer();
    accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
    accelerationStructureCreateInfo.type = accelerationStructureType;
    accelerationStructureCreateInfo.deviceAddress = 0;

    if(vulkan_context->extensionFuncs.vkCreateAccelerationStructureKHR(vulkan_context->getDevice()->getLogicalDevice(), &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS)
    {
        throw std::runtime_error("[AccelerationStructure::createAccelerationStructure()] ERROR - failed to creates acceleration structure!");
    }
    
    VkAccelerationStructureDeviceAddressInfoKHR accelerationStructureDeviceAddressInfo{};
    accelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationStructureDeviceAddressInfo.accelerationStructure = accelerationStructure;

    accelerationStructureDeviceAddress = vulkan_context->extensionFuncs.vkGetAccelerationStructureDeviceAddressKHR(
        vulkan_context->getDevice()->getLogicalDevice(), 
        &accelerationStructureDeviceAddressInfo);
    
    accelerationStructureScratchMem.resize(accelerationStructureBuildSizesInfo.buildScratchSize);

    accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = accelerationStructureScratchMem.getBuffer()->getBufferDeviceAddress();

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
