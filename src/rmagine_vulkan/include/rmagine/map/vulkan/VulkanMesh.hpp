#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "../../util/VulkanContext.hpp"
#include "../../util/VulkanUtil.hpp"
#include "AccelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "VulkanEntity.hpp"
// #include "../../../rmagine_core/map/AssimpIO.hpp"
#include "../../../rmagine_core/math/Types.hpp"



namespace rmagine
{

class VulkanMesh : public VulkanEntity
{
private:

public:
    Memory<Point, VULKAN_DEVICE_LOCAL>    vertices;
    Memory<Face, VULKAN_DEVICE_LOCAL>     faces;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   face_normals;
    Memory<Vector, VULKAN_DEVICE_LOCAL>   vertex_normals;

    VkTransformMatrixKHR transformMatrix{};

    VulkanMesh();

    virtual ~VulkanMesh();

    VulkanMesh(const VulkanMesh&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    virtual void apply();

    virtual void commit();

    virtual unsigned int depth() const;

    void computeFaceNormals();
};

using VulkanMeshPtr = std::shared_ptr<VulkanMesh>;



VulkanMeshPtr make_vulkan_mesh(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram);

// VulkanMeshPtr make_vulkan_mesh(const aiMesh* amesh);

} // namespace rmagine
