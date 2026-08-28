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

    // re-reads current transform/AS-reference of every instance into m_asInstances(_ram)
    void updateInstanceData(std::map<unsigned int, VulkanGeometryPtr>& geometries);

    // fills geometry/build-range descriptors pointing at m_asInstances - shared by BUILD and UPDATE
    void makeGeometryInput(
        size_t n_instances,
        std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys,
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos);

public:
    Memory<VkDeviceAddress, RAM> m_asInstancesDescriptions_ram;
    Memory<VkDeviceAddress, DEVICE_LOCAL_VULKAN> m_asInstancesDescriptions;

    TopLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries);
    ~TopLevelAccelerationStructure();

    /**
     * @brief refit this TLAS in place for the given geometries
     *
     * only valid when the number/identity of instances is the same as during the last
     * build/update (e.g. only instance transforms changed) - if instances were added or
     * removed, construct a new TopLevelAccelerationStructure instead.
     */
    void update(std::map<unsigned int, VulkanGeometryPtr>& geometries);
};

} // namespace rmagine
