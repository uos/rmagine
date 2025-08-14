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

#include <rmagine/math/types.h>
#include "vulkan_definitions.hpp"
#include "VulkanGeometry.hpp"
#include "accelerationStructure/BottomLevelGeometryInstance.hpp"



namespace rmagine
{

class VulkanInst : VulkanGeometry
{
protected:
    VulkanScenePtr m_scene;

    BottomLevelGeometryInstancePtr bottomLevelGeometryInstance = nullptr;

public:
    using Base = VulkanGeometry;

    VulkanInst();

    virtual ~VulkanInst();


    void set(VulkanScenePtr geom);
    VulkanScenePtr scene() const;

    void setId(unsigned int id);
    unsigned int id() const;

    void disable();
    void enable();

    virtual VulkanGeometryType type() const
    {
        return VulkanGeometryType::INSTANCE;
    }
};

using VulkanInstPtr = std::shared_ptr<VulkanInst>;

} // namespace rmagine
