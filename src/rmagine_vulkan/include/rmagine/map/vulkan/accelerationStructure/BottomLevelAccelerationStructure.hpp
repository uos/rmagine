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
private:
    // re-reads current vertex/face/transform buffer addresses of every mesh, filling both
    // m_meshDescriptions(_ram) and the geometry/build-range descriptors - shared by BUILD and UPDATE
    void makeGeometryInput(
        std::map<unsigned int, VulkanGeometryPtr>& geometries,
        std::vector<VkAccelerationStructureGeometryKHR>& accelerationStructureGeometrys,
        std::vector<VkAccelerationStructureBuildRangeInfoKHR>& accelerationStructureBuildRangeInfos);

public:
    Memory<MeshDescription, RAM> m_meshDescriptions_ram;
    Memory<MeshDescription, DEVICE_LOCAL_VULKAN> m_meshDescriptions;

    BottomLevelAccelerationStructure(std::map<unsigned int, VulkanGeometryPtr>& geometries);
    ~BottomLevelAccelerationStructure();

    /**
     * @brief refit this BLAS in place for the given geometries
     *
     * only valid when the number/identity of meshes and each mesh's vertex/face count are
     * the same as during the last build/update (e.g. only vertex positions or a mesh's
     * pre-transform changed) - if meshes were added/removed, construct a new
     * BottomLevelAccelerationStructure instead.
     */
    void update(std::map<unsigned int, VulkanGeometryPtr>& geometries);
};

} // namespace rmagine
