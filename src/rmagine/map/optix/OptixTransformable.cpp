#include "rmagine/map/optix/OptixTransformable.hpp"

#include <rmagine/math/linalg.h>

namespace rmagine
{

OptixTransformable::OptixTransformable()
{
    m_T.setIdentity();
    m_S = {1.0, 1.0, 1.0};
    m_changed = true;
}

void OptixTransformable::setTransform(const Transform& T)
{
    m_T = T;
    m_changed = true;
}

void OptixTransformable::setTransform(const Matrix4x4& T)
{
    // scale?
    Transform T2;
    T2.set(T);
    setTransform(T2);
    m_changed = true;
}

void OptixTransformable::setTransformAndScale(const Matrix4x4& M)
{
    decompose(M, m_T, m_S);
    m_changed = true;
}

Transform OptixTransformable::transform() const
{
    return m_T;
}

void OptixTransformable::setScale(const Vector3& S)
{
    m_S = S;
    m_changed = true;
}

Vector3 OptixTransformable::scale() const
{
    return m_S;
}

Matrix4x4 OptixTransformable::matrix() const
{
    return compose(m_T, m_S);
}

} // namespace rmagine