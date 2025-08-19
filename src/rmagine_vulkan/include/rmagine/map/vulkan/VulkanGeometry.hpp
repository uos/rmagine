#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_set>

#include <vulkan/vulkan.h>

#include <rmagine/math/types.h>
#include "vulkan_definitions.hpp"
#include "VulkanEntity.hpp"
#include "VulkanTransformable.hpp"



namespace rmagine
{

class VulkanGeometry : public VulkanEntity, public VulkanTransformable
{
protected:
    std::unordered_set<VulkanSceneWPtr> m_parents;

    VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};

public:
    VulkanGeometry() : VulkanEntity(), VulkanTransformable() {}

    virtual ~VulkanGeometry() {}


    virtual VulkanGeometryType type() const = 0;

    virtual unsigned int depth() const = 0;

    virtual void commit() = 0;

    // handle parents
    void cleanupParents();
    std::unordered_set<VulkanScenePtr> parents() const;
    bool removeParent(VulkanScenePtr parent);
    bool hasParent(VulkanScenePtr parent) const;
    void addParent(VulkanScenePtr parent);

    VulkanScenePtr makeScene();
    VulkanInstPtr instantiate();

    const VkAccelerationStructureGeometryKHR& getASGeometry() const;
    const VkAccelerationStructureBuildRangeInfoKHR& getASBuildRangeInfo() const;
};

using VulkanGeometryPtr = std::shared_ptr<VulkanGeometry>;

} // namespace rmagine
