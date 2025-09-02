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
#include <rmagine/util/vulkan/general/CommandBuffer.hpp>
#include <rmagine/map/vulkan/vulkan_definitions.hpp>
#include <rmagine/types/MemoryVulkan.hpp>



namespace rmagine
{

class AccelerationStructure : public std::enable_shared_from_this<AccelerationStructure>
{
private:
    VulkanContextPtr vulkan_context = nullptr;
    CommandBufferPtr commandBuffer = nullptr;

    VkAccelerationStructureTypeKHR accelerationStructureType = VkAccelerationStructureTypeKHR::VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;

    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkDeviceAddress accelerationStructureDeviceAddress = 0;
    // for acceleration structure
    BufferPtr accelerationStructureBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureDeviceMemory = nullptr;
    // for building acceleration structure
    BufferPtr accelerationStructureScratchBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureScratchDeviceMemory = nullptr;
    
public:
    AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType);

    virtual ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;


    VkDeviceAddress getDeviceAddress();

    VkAccelerationStructureKHR* getAcceleratiionStructurePtr();

    size_t getID();

    template<typename T>
    inline std::shared_ptr<T> this_shared()
    {
        return std::dynamic_pointer_cast<T>(shared_from_this());
    }

protected:
    void createAccelerationStructure(
        std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, 
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos);
};

using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;

} // namespace rmagine
