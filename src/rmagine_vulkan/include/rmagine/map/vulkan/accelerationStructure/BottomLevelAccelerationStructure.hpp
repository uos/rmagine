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

    void createAccelerationStructure(std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys, std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos);
};

using BottomLevelAccelerationStructurePtr = std::shared_ptr<BottomLevelAccelerationStructure>;

} // namespace rmagine
