#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryVulkan.hpp>
#include "vulkan_definitions.hpp"
#include "VulkanGeometry.hpp"



namespace rmagine
{

class VulkanInst : public VulkanGeometry
{
protected:
    VulkanScenePtr m_scene;

    VkAccelerationStructureInstanceKHR* m_data;// is a pointer just to be consistent with the OptixInst

public:
    using Base = VulkanGeometry;

    VulkanInst();

    virtual ~VulkanInst();


    void set(VulkanScenePtr geom);
    VulkanScenePtr scene() const;

    virtual void apply();
    virtual void commit();
    virtual unsigned int depth() const;

    void setId(unsigned int id);
    unsigned int id() const;

    void disable();
    void enable();

    virtual VulkanGeometryType type() const
    {
        return VulkanGeometryType::INSTANCE;
    }

    const VkAccelerationStructureInstanceKHR* data() const;
};

using VulkanInstPtr = std::shared_ptr<VulkanInst>;

} // namespace rmagine
