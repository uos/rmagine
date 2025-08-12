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

#include "../util/VulkanContext.hpp"
#include "../util/VulkanUtil.hpp"
#include "mapComponents/VulkanScene.hpp"
#include "../../rmagine_core/map/Map.hpp"
// #include "../../rmagine_core/map/AssimpIO.hpp"



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
