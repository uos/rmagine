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

#include "../../util/MemoryVulkan.hpp"



namespace rmagine
{

class VulkanTransformable
{
protected:
    Transform m_T;
    Vector3 m_S;
    
public:
    bool m_changed;

    VulkanTransformable(/* args */);

    virtual ~VulkanTransformable();
    

    void setTransform(const Transform& T);

    Transform transform() const;

    void setScale(const Vector3& S);

    Vector3 scale() const;

    // void setTransform(const Matrix4x4& T);

    // void setTransformAndScale(const Matrix4x4& T);

    // Matrix4x4 matrix() const;

    bool changed() const;
};

using VulkanTransformablePtr = std::shared_ptr<VulkanTransformable>;

} // namespace rmagine
