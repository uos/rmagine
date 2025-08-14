#include "rmagine/map/vulkan/VulkanTransformable.hpp"



namespace rmagine
{

VulkanTransformable::VulkanTransformable(/* args */)
{
    m_T.setIdentity();
    m_S = {1.0, 1.0, 1.0};
    m_changed = true;
}

void VulkanTransformable::setTransform(const Transform& T)
{
    m_T = T;
    m_changed = true;
}

Transform VulkanTransformable::transform() const
{
    return m_T;
}

void VulkanTransformable::setScale(const Vector3& S)
{
    m_S = S;
    m_changed = true;
}

Vector3 VulkanTransformable::scale() const
{
    return m_S;
}

void VulkanTransformable::setTransform(const Matrix4x4& T)
{
    Transform T2;
    T2.set(T);
    setTransform(T2);
    m_changed = true;
}

void VulkanTransformable::setTransformAndScale(const Matrix4x4& M)
{
    decompose(M, m_T, m_S);
    m_changed = true;
}

Matrix4x4 VulkanTransformable::matrix() const
{
    return compose(m_T, m_S);
}

bool VulkanTransformable::changed() const
{
    return m_changed;
}

} // namespace rmagine
