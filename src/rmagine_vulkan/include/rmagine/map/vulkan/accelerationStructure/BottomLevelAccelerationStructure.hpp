#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include <vulkan/vulkan.h>

#include "AccelerationStructure.hpp"
#include "../../../util/MemoryVulkan.hpp"
#include "../../../util/VulkanUtil.hpp"



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
