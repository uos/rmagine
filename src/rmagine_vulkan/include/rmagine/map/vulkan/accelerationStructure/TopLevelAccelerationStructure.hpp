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
    Memory<VkAccelerationStructureInstanceKHR, DEVICE_LOCAL_VULKAN> m_asInstances;

public:
    Memory<VkDeviceAddress, RAM> m_asInstancesDescriptions_ram;
    Memory<VkDeviceAddress, DEVICE_LOCAL_VULKAN> m_asInstancesDescriptions;
    
    TopLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries);
    ~TopLevelAccelerationStructure();
};

} // namespace rmagine
