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
    Memory<VkAccelerationStructureInstanceKHR, RAM> instanceMemory_ram;

public:
    BottomLevelAccelerationStructureInstance() : 
    instanceMemory(1, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR),
    instanceMemory_ram(1) {}
    
    ~BottomLevelAccelerationStructureInstance() {}

    BottomLevelAccelerationStructureInstance(const BottomLevelAccelerationStructureInstance&) = delete;

    
    void createBottomLevelAccelerationStructureInstance(BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure);

    void createBottomLevelAccelerationStructureInstance(VkTransformMatrixKHR transformMatrix, uint32_t mask, BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure);

    void updateTransformMatrix(VkTransformMatrixKHR transformMatrix);

    void updateMask(uint32_t mask);

    void updateTransformMatrixAndMask(VkTransformMatrixKHR transformMatrix, uint32_t mask);

    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>& getInstanceMemory();

    void cleanup();
};

using BottomLevelAccelerationStructureInstancePtr = std::shared_ptr<BottomLevelAccelerationStructureInstance>;

} // namespace rmagine
