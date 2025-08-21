#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "AccelerationStructure.hpp"



namespace rmagine
{

class BottomLevelAccelerationStructure : public AccelerationStructure
{
public:
    Memory<MeshDescription, RAM> m_meshDescriptions_ram;
    Memory<MeshDescription, VULKAN_DEVICE_LOCAL> m_meshDescriptions;

    BottomLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries);
    ~BottomLevelAccelerationStructure();
};

} // namespace rmagine
