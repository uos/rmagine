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
#include "VulkanGeometry.hpp"



namespace rmagine
{

class VulkanMesh : public VulkanGeometry
{
protected:
    Memory<VkTransformMatrixKHR, RAM>   transformMatrix_ram;

public:
    using Base = VulkanGeometry;

    Memory<VkTransformMatrixKHR, VULKAN_DEVICE_LOCAL>   transformMatrix;

    Memory<Point, VULKAN_DEVICE_LOCAL>    vertices;
    Memory<Face, VULKAN_DEVICE_LOCAL>     faces;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   face_normals;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   vertex_normals;

    VulkanMesh();

    virtual ~VulkanMesh();


    virtual void apply();

    virtual void commit();

    virtual unsigned int depth() const;

    virtual VulkanGeometryType type() const 
    {
        return VulkanGeometryType::MESH;
    }

    void computeFaceNormals();
};

using VulkanMeshPtr = std::shared_ptr<VulkanMesh>;



VulkanMeshPtr make_vulkan_mesh(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram);

VulkanMeshPtr make_vulkan_mesh(const aiMesh* amesh);

} // namespace rmagine
