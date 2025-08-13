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

#include <rmagine/math/types.h>
#include "VulkanEntity.hpp"
#include "VulkanTransformable.hpp"



namespace rmagine
{

class VulkanGeometry : public VulkanEntity, public VulkanTransformable
{
protected:
    // std::unordered_set<VulkanSceneWPtr> m_parents;

public:
    VulkanGeometry(/* args */);
    virtual ~VulkanGeometry();
};

using VulkanGeometryPtr = std::shared_ptr<VulkanGeometry>;

} // namespace rmagine
