#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/map/AssimpIO.hpp>
#include <rmagine/math/types.h>
#include <rmagine/math/assimp_conversions.h>
#include <rmagine/types/mesh_types.h>
#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include "vulkan_definitions.hpp"
#include "accelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "VulkanGeometry.hpp"



namespace rmagine
{

class VulkanMesh : public VulkanGeometry
{
private:
    BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure = nullptr;

public:
    Memory<Point, VULKAN_DEVICE_LOCAL>    vertices;
    Memory<Face, VULKAN_DEVICE_LOCAL>     faces;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   face_normals;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   vertex_normals;

    VulkanMesh();

    virtual ~VulkanMesh();

    VulkanMesh(const VulkanMesh&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    virtual void apply();

    virtual void commit();

    virtual unsigned int depth() const;

    virtual VulkanGeometryType type() const 
    {
        return VulkanGeometryType::MESH;
    }

    // void computeFaceNormals();
};

using VulkanMeshPtr = std::shared_ptr<VulkanMesh>;



VulkanMeshPtr make_vulkan_mesh(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram);

// VulkanMeshPtr make_vulkan_mesh(const aiMesh* amesh);

} // namespace rmagine
