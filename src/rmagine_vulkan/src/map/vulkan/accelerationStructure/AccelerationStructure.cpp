#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"
#include <iostream>



namespace rmagine
{

AccelerationStructure::AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType) : 
    accelerationStructureType(accelerationStructureType),
    vulkan_context(get_vulkan_context_weak()),
    commandBuffer(new CommandBuffer(vulkan_context)),
    accelerationStructureMem(0, VulkanMemoryUsage::Usage_AccelerationStructure)
{
    
}

AccelerationStructure::~AccelerationStructure()
{
    if(accelerationStructure != VK_NULL_HANDLE)
    {
        commandBuffer.reset();

        vulkan_context->extensionFuncs.vkDestroyAccelerationStructureKHR(vulkan_context->getDevice()->getLogicalDevice(), accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
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

    std::cout << "[DIAG-AS] type=" << accelerationStructureType
              << " geomCount=" << accelerationStructureGeometrys.size()
              << " maxPrim[0]=" << (maxPrimitiveCountList.empty() ? -1 : (int)maxPrimitiveCountList[0])
              << " asSize=" << accelerationStructureBuildSizesInfo.accelerationStructureSize
              << " buildScratch=" << accelerationStructureBuildSizesInfo.buildScratchSize
              << " geomType[0]=" << (accelerationStructureGeometrys.empty() ? -1 : (int)accelerationStructureGeometrys[0].geometryType)
              << std::endl;

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

    std::cout << "[DIAG-AS] handle=" << (void*)accelerationStructure
              << " deviceAddress=" << accelerationStructureDeviceAddress << std::endl;

    // for building acceleration structure
    Memory<char, DEVICE_LOCAL_VULKAN> accelerationStructureScratchMem(accelerationStructureBuildSizesInfo.buildScratchSize);

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


size_t AccelerationStructure::getID() const
{
    return accelerationStructureMem.getID();
}


size_t AccelerationStructure::getSize() const
{
    return accelerationStructureMem.size();
}

} // namespace rmagine
