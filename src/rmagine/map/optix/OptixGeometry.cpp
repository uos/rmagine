#include "rmagine/map/optix/OptixGeometry.hpp"
#include <rmagine/math/linalg.h>

namespace rmagine
{

OptixGeometry::OptixGeometry(OptixContextPtr context)
:m_ctx(context)
{
    m_T.setIdentity();
    m_S = {1.0, 1.0, 1.0};
    std::cout << "[OptixGeometry::OptixGeometry()] constructed." << std::endl;
}

OptixGeometry::~OptixGeometry()
{
    if(m_as)
    {
        cudaFree( reinterpret_cast<void*>( m_as->buffer ) );
    }
    std::cout << "[OptixGeometry::~OptixGeometry()] destroyed." << std::endl;
}

OptixAccelerationStructurePtr OptixGeometry::handle()
{
    return m_as;
}

void OptixGeometry::apply()
{

}

void OptixGeometry::setTransform(const Transform& T)
{
    m_T = T;
}

void OptixGeometry::setTransform(const Matrix4x4& T)
{
    // scale?
    Transform T2;
    T2.set(T);
    setTransform(T2);
}

void OptixGeometry::setTransformAndScale(const Matrix4x4& M)
{
    decompose(M, m_T, m_S);
}

Transform OptixGeometry::transform() const
{
    return m_T;
}

void OptixGeometry::setScale(const Vector3& S)
{
    m_S = S;
}

Vector3 OptixGeometry::scale() const
{
    return m_S;
}

Matrix4x4 OptixGeometry::matrix() const
{
    return compose(m_T, m_S);
}

} // namespace rmagine