#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include <vulkan/vulkan.h>

#include <rmagine/util/vulkan/memory/Buffer.hpp>
#include <rmagine/util/vulkan/memory/DeviceMemory.hpp>



namespace rmagine
{

class AccelerationStructure
{
protected:
    DevicePtr device = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkDeviceAddress accelerationStructureDeviceAddress = uint64_t(~0);
    // for acceleration structure
    BufferPtr accelerationStructureBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureDeviceMemory = nullptr;
    // for building acceleration structure
    BufferPtr accelerationStructureScratchBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureScratchDeviceMemory = nullptr;
    
public:
    AccelerationStructure();
    
    AccelerationStructure(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr);

    ~AccelerationStructure() {}

    AccelerationStructure(const AccelerationStructure&) = delete;
    

    VkDeviceAddress getDeviceAddress();

    VkAccelerationStructureKHR* getAcceleratiionStructurePtr();

    void cleanup();

protected:
    void createAccelerationStructureBufferAndDeviceMemory(std::vector<uint32_t> maxPrimitiveCountList, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo, VkAccelerationStructureTypeKHR accelerationStructureType);

    void buildAccelerationStructure(uint32_t primitiveCount, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo);
};

} // namespace rmagine
