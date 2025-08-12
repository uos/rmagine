#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include <vulkan/vulkan.h>

#include "BottomLevelAccelerationStructure.hpp"
#include "../../../util/MemoryVulkan.hpp"



namespace rmagine
{

class BottomLevelGeometryInstance
{
private:
    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL> geometryInstanceMemory;

public:
    BottomLevelGeometryInstance() 
        : geometryInstanceMemory(Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>(1, 
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)) {}
    
    ~BottomLevelGeometryInstance() {}

    BottomLevelGeometryInstance(const BottomLevelGeometryInstance&) = delete;

    
    void createBottomLevelAccelerationStructureInstance(VkTransformMatrixKHR transformMatrix, BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure);

    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>& getGeometryInstanceMemory();

    void cleanup();
};

using BottomLevelGeometryInstancePtr = std::shared_ptr<BottomLevelGeometryInstance>;

} // namespace rmagine
