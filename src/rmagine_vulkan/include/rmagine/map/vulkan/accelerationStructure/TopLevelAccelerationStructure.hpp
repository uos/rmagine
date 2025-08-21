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

class TopLevelAccelerationStructure : public AccelerationStructure
{
private:
    Memory<VkAccelerationStructureInstanceKHR, RAM> m_asInstances_ram;
    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL> m_asInstances;

public:
    Memory<VkDeviceAddress, RAM> m_blasMeshDescriptions_ram;
    Memory<VkDeviceAddress, VULKAN_DEVICE_LOCAL> m_blasMeshDescriptions;
    
    TopLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries);
    ~TopLevelAccelerationStructure();
};

} // namespace rmagine
