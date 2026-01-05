#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/math/types.h>
#include <rmagine/math/linalg.h>



namespace rmagine
{

class VulkanTransformable
{
protected:
    Transform m_T;
    Vector3 m_S;
    
public:
    bool m_changed;

    VulkanTransformable();

    virtual ~VulkanTransformable();
    

    void setTransform(const Transform& T);

    Transform transform() const;

    void setScale(const Vector3& S);

    Vector3 scale() const;

    void setTransform(const Matrix4x4& T);

    void setTransformAndScale(const Matrix4x4& T);

    Matrix4x4 matrix() const;

    bool changed() const;

    virtual void apply() = 0;
};

using VulkanTransformablePtr = std::shared_ptr<VulkanTransformable>;

} // namespace rmagine
