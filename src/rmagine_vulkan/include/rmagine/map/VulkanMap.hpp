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
#include <rmagine/util/VulkanContextUtil.hpp>
#include "vulkan/VulkanScene.hpp"



namespace rmagine
{

class VulkanMap : public Map, public VulkanEntity
{
private:
    VulkanScenePtr m_scene;

public:
    VulkanMap();

    VulkanMap(VulkanScenePtr scene);

    virtual ~VulkanMap();


    void setScene(VulkanScenePtr scene);

    VulkanScenePtr scene() const;
};

using VulkanMapPtr = std::shared_ptr<VulkanMap>;



VulkanMapPtr import_vulkan_map(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram);

VulkanMapPtr import_vulkan_map(const std::string& meshfile);

} // namespace rmagine
