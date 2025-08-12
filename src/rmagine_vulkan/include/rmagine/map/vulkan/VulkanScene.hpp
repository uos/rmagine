#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>
#include <unordered_set>

#include <vulkan/vulkan.h>

#include "../../util/VulkanContext.hpp"
#include "../../util/VulkanUtil.hpp"
#include "AccelerationStructure/BottomLevelAccelerationStructure.hpp"
#include "AccelerationStructure/TopLevelAccelerationStructure.hpp"
#include "AccelerationStructure/BottomLevelGeometryInstance.hpp"
#include "VulkanMesh.hpp"
#include "VulkanEntity.hpp"
// #include "VulkanGeometry.hpp"
// #include "VulkanTransformable.hpp"
// #include "VulkanInst.hpp"
// #include "../../../rmagine_core/map/AssimpIO.hpp"



namespace rmagine
{

class VulkanScene : public VulkanEntity
{
private:
    //memory
    uint32_t numVerticies = 0;
    uint32_t numTriangles = 0;
    Memory<float, VULKAN_DEVICE_LOCAL> vertexMem;
    Memory<uint32_t, VULKAN_DEVICE_LOCAL> indexMem;

    //acceleration structure
    BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure = nullptr;//TODO: multiple of these
    BottomLevelGeometryInstancePtr bottomLevelGeometryInstance = nullptr;//TODO: multiple of these
    TopLevelAccelerationStructurePtr topLevelAccelerationStructure = nullptr;

public:
    VulkanScene()
    {

    }

    VulkanScene(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram) : 
        vertexMem(Memory<float, VULKAN_DEVICE_LOCAL>(vertexMem_ram.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)),
        indexMem(Memory<uint32_t, VULKAN_DEVICE_LOCAL>(indexMem_ram.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)),
        bottomLevelAccelerationStructure(new BottomLevelAccelerationStructure()), 
        bottomLevelGeometryInstance(new BottomLevelGeometryInstance),
        topLevelAccelerationStructure(new TopLevelAccelerationStructure()) 
    {
        createScene(vertexMem_ram, indexMem_ram);
    }

    ~VulkanScene()
    {
        std::cout << "destroying VulkanScene" << std::endl;
        cleanup();
    }

    VulkanScene(const VulkanScene&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void commit();

    void cleanup();

    Memory<float, VULKAN_DEVICE_LOCAL>& getVertexMem();

    Memory<uint32_t, VULKAN_DEVICE_LOCAL>& getIndexMem();

    TopLevelAccelerationStructurePtr getTopLevelAccelerationStructure();

private:
    void createScene(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram);
};

using VulkanScenePtr = std::shared_ptr<VulkanScene>;



VulkanScenePtr make_vulkan_scene(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram);

// VulkanScenePtr make_vulkan_scene(const aiScene* ascene);

} // namespace rmagine
