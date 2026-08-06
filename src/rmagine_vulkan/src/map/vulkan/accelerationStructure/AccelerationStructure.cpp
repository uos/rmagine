#include "rmagine/map/vulkan/accelerationStructure/AccelerationStructure.hpp"
#include "rmagine/util/VulkanContext.hpp"



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
    std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos,
    VkBuildAccelerationStructureModeKHR mode)
{
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
    accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accelerationStructureBuildGeometryInfo.type = accelerationStructureType;
    // always allow future refits, even on the very first (mode == BUILD) call
    accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    accelerationStructureBuildGeometryInfo.mode = mode;
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

    VkDeviceSize scratchSize;

    if(mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
    {
        // refit in place: same handle, same backing buffer, primitive counts unchanged
        // since the last BUILD - only vertex/instance/transform data changed.
        if(accelerationStructure == VK_NULL_HANDLE)
        {
            throw std::runtime_error("[AccelerationStructure::createAccelerationStructure()] ERROR - cannot update an acceleration structure that was never built!");
        }

        accelerationStructureBuildGeometryInfo.srcAccelerationStructure = accelerationStructure;
        accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
        scratchSize = accelerationStructureBuildSizesInfo.updateScratchSize;
    }
    else
    {
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

        accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
        scratchSize = accelerationStructureBuildSizesInfo.buildScratchSize;
    }

    // for building/updating acceleration structure
    Memory<char, DEVICE_LOCAL_VULKAN> accelerationStructureScratchMem(scratchSize);
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
