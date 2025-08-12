#include "AccelerationStructure.hpp"
#include "../../../util/VulkanContext.hpp"



namespace rmagine
{

AccelerationStructure::AccelerationStructure() : 
    device(get_vulkan_context()->getDevice()), 
    extensionFunctionsPtr(get_vulkan_context()->getExtensionFunctionsPtr()) {}

AccelerationStructure::AccelerationStructure(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr) : 
    device(device), extensionFunctionsPtr(extensionFunctionsPtr) {}


void AccelerationStructure::createAccelerationStructureBufferAndDeviceMemory(std::vector<uint32_t> maxPrimitiveCountList, 
    VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
    VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo, 
    VkAccelerationStructureTypeKHR accelerationStructureType)
{
    if(accelerationStructureBuffer != nullptr)
    {
        throw std::runtime_error("acceleration structure buffer has already been created!");
    }

    extensionFunctionsPtr->pvkGetAccelerationStructureBuildSizesKHR(
        device->getLogicalDevice(), 
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, 
        &accelerationStructureBuildGeometryInfo, 
        maxPrimitiveCountList.data(), 
        &accelerationStructureBuildSizesInfo);

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

    if(extensionFunctionsPtr->pvkCreateAccelerationStructureKHR(device->getLogicalDevice(), &accelerationStructureCreateInfo, nullptr, &accelerationStructure) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to creates acceleration structure!");
    }
    
    VkAccelerationStructureDeviceAddressInfoKHR accelerationStructureDeviceAddressInfo{};
    accelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationStructureDeviceAddressInfo.accelerationStructure = accelerationStructure;

    accelerationStructureDeviceAddress = 
        extensionFunctionsPtr->pvkGetAccelerationStructureDeviceAddressKHR(device->getLogicalDevice(), &accelerationStructureDeviceAddressInfo);
}

void AccelerationStructure::buildAccelerationStructure(uint32_t primitiveCount, 
    VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
    VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo)
{
    if(accelerationStructureBuffer == nullptr)
    {
        throw std::runtime_error("acceleration structure buffer has to get created before acceleration structure can be build!");
    }

    accelerationStructureScratchBuffer = std::make_shared<Buffer>(
        accelerationStructureBuildSizesInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    accelerationStructureScratchDeviceMemory = std::make_shared<DeviceMemory>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, accelerationStructureScratchBuffer);

    accelerationStructureBuildGeometryInfo.dstAccelerationStructure = accelerationStructure;
    accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = accelerationStructureScratchBuffer->getBufferDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount = primitiveCount;

    const VkAccelerationStructureBuildRangeInfoKHR* accelerationStructureBuildRangeInfos = &accelerationStructureBuildRangeInfo;

    get_vulkan_context()->getDefaultCommandBuffer()->recordBuildingASToCommandBuffer(accelerationStructureBuildGeometryInfo, accelerationStructureBuildRangeInfos);
    get_vulkan_context()->getDefaultCommandBuffer()->submitRecordedCommandAndWait();

    std::cout << "acceleration structure has been build" << std::endl;
}

VkDeviceAddress AccelerationStructure::getDeviceAddress()
{
    return accelerationStructureDeviceAddress;
}

VkAccelerationStructureKHR* AccelerationStructure::getAcceleratiionStructurePtr()
{
    return &accelerationStructure;
}

void AccelerationStructure::cleanup()
{
    if(accelerationStructure != VK_NULL_HANDLE)
    {
        accelerationStructureDeviceMemory->cleanup();
        accelerationStructureBuffer->cleanup();
        accelerationStructureScratchDeviceMemory->cleanup();
        accelerationStructureScratchBuffer->cleanup();

        extensionFunctionsPtr->pvkDestroyAccelerationStructureKHR(device->getLogicalDevice(), accelerationStructure, nullptr);
        accelerationStructure = VK_NULL_HANDLE;
    }
}

} // namespace rmagine
