#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "BottomLevelAccelerationStructure.hpp"



namespace rmagine
{

class BottomLevelAccelerationStructureInstance
{
private:
    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL> instanceMemory;

public:
    BottomLevelAccelerationStructureInstance() 
        : instanceMemory(Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>(1, 
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)) {}
    
    ~BottomLevelAccelerationStructureInstance() {}

    BottomLevelAccelerationStructureInstance(const BottomLevelAccelerationStructureInstance&) = delete;

    
    void createBottomLevelAccelerationStructureInstance(VkTransformMatrixKHR transformMatrix, BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure);

    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>& getInstanceMemory();

    void cleanup();
};

using BottomLevelAccelerationStructureInstancePtr = std::shared_ptr<BottomLevelAccelerationStructureInstance>;

} // namespace rmagine
