#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include "AccelerationStructure.hpp"



namespace rmagine
{

class BottomLevelAccelerationStructure final : public AccelerationStructure
{
private:
    /* data */

public:
    BottomLevelAccelerationStructure() : AccelerationStructure() {}
    BottomLevelAccelerationStructure(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr) :
        AccelerationStructure(device, extensionFunctionsPtr) {}
    ~BottomLevelAccelerationStructure(){}

    BottomLevelAccelerationStructure(const BottomLevelAccelerationStructure&) = delete;

    void createAccelerationStructure(uint32_t numVerticies, Memory<float, VULKAN_DEVICE_LOCAL>& vertexMem, uint32_t numTriangles, Memory<uint32_t, VULKAN_DEVICE_LOCAL>& indexMem);
};

using BottomLevelAccelerationStructurePtr = std::shared_ptr<BottomLevelAccelerationStructure>;

} // namespace rmagine
