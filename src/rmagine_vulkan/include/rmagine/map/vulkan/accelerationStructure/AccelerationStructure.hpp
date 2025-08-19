#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/vulkan/memory/Buffer.hpp>
#include <rmagine/util/vulkan/memory/DeviceMemory.hpp>



namespace rmagine
{

class AccelerationStructure : public std::enable_shared_from_this<AccelerationStructure>
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

    virtual ~AccelerationStructure() {}

    AccelerationStructure(const AccelerationStructure&) = delete;
    

    VkDeviceAddress getDeviceAddress();

    VkAccelerationStructureKHR* getAcceleratiionStructurePtr();

    void cleanup();

    template<typename T>
    inline std::shared_ptr<T> this_shared()
    {
        return std::dynamic_pointer_cast<T>(shared_from_this());
    }

protected:
    void createAccelerationStructureBufferAndDeviceMemory(std::vector<uint32_t> maxPrimitiveCountList, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo, VkAccelerationStructureTypeKHR accelerationStructureType);

    void buildAccelerationStructure(std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo);
};

using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;

} // namespace rmagine
