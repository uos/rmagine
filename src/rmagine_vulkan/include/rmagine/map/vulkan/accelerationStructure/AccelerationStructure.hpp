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
#include "vulkan_definitions.hpp"



namespace rmagine
{

enum AccelerationStructureType
{
    TOP_LEVEL,
    BOTTOM_LEVEL
};



class AccelerationStructure : public std::enable_shared_from_this<AccelerationStructure>
{
private:
    VkAccelerationStructureTypeKHR accelerationStructureType = VkAccelerationStructureTypeKHR::VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;

    DevicePtr device = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkDeviceAddress accelerationStructureDeviceAddress = 0;
    // for acceleration structure
    BufferPtr accelerationStructureBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureDeviceMemory = nullptr;
    // for building acceleration structure
    BufferPtr accelerationStructureScratchBuffer = nullptr;
    DeviceMemoryPtr accelerationStructureScratchDeviceMemory = nullptr;

    size_t asID = 0;
    
public:
    AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType);
    
    AccelerationStructure(VkAccelerationStructureTypeKHR accelerationStructureType, DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr);

    virtual ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    

    void createAccelerationStructure(
        std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, 
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos);

    VkDeviceAddress getDeviceAddress();

    VkAccelerationStructureKHR* getAcceleratiionStructurePtr();

    size_t getID();

private:
    void createAccelerationStructureBufferAndDeviceMemory(
        std::vector<uint32_t>& maxPrimitiveCountList, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo);

    void buildAccelerationStructure(
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos, 
        VkAccelerationStructureBuildGeometryInfoKHR& accelerationStructureBuildGeometryInfo, 
        VkAccelerationStructureBuildSizesInfoKHR& accelerationStructureBuildSizesInfo);

    void cleanup();

private:
    static size_t asIDcounter;

    static size_t getNewAsID();

    static VkAccelerationStructureBuildRangeInfoKHR GetASBuildRange(VulkanScenePtr scene);

    static VkAccelerationStructureBuildRangeInfoKHR GetASBuildRange(VulkanMeshPtr mesh);

    static VkAccelerationStructureGeometryKHR GetASGeometry(VulkanScenePtr scene);

    static VkAccelerationStructureGeometryKHR GetASGeometry(VulkanMeshPtr mesh);
};

using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;

} // namespace rmagine
