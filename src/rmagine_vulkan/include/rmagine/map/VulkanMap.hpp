#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/map/Map.hpp>
#include <rmagine/map/AssimpIO.hpp>
#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/VulkanUtil.hpp>
#include "vulkan/VulkanScene.hpp"



namespace rmagine
{

class VulkanMap : public Map
{
private:
    VulkanContextPtr vulkan_context = nullptr;
    
    VulkanScenePtr m_scene;

public:
    VulkanMap(VulkanScenePtr scene) : vulkan_context(get_vulkan_context()), m_scene(scene)
    {}

    ~VulkanMap()
    {
        std::cout << "destroying VulkanMap" << std::endl;
        cleanup();
    }

    VulkanMap(const VulkanMap&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    VulkanScenePtr scene() const;

private:
    void cleanup();
};

using VulkanMapPtr = std::shared_ptr<VulkanMap>;



VulkanMapPtr import_vulkan_map(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram);

// VulkanMapPtr import_vulkan_map(const std::string& meshfile);

} // namespace rmagine
