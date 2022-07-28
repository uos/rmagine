
#include "rmagine/map/embree/EmbreeGeometry.hpp"
#include <iostream>
#include <Eigen/Dense>

namespace rmagine
{

void decompose(const Matrix4x4& M, Transform& T, Vector3& scale)
{
    Eigen::Matrix4f Meig;
    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            Meig(i, j) = M(i, j);
        }
    }

    Eigen::Affine3f A;
    A.matrix() = Meig;


    Eigen::Matrix3f Reig;
    Eigen::Matrix3f Seig;
    A.computeRotationScaling(&Reig, &Seig);
    
    Eigen::Vector3f t = A.translation();
    
    Matrix3x3 R;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            R(i,j) = Reig(i,j);
        }
    }

    T.t = {t.x(), t.y(), t.z()};
    T.R.set(R);

    scale.x = Seig(0,0);
    scale.y = Seig(1,1);
    scale.z = Seig(2,2);
}

EmbreeGeometry::EmbreeGeometry(EmbreeDevicePtr device)
:m_device(device)
,m_S{1.0,1.0,1.0}
{
    m_T.setIdentity();
    std::cout << "[EmbreeGeometry::EmbreeGeometry()] constructed." << std::endl;
}

EmbreeGeometry::~EmbreeGeometry()
{
    release();
    std::cout << "[EmbreeGeometry::~EmbreeGeometry()] destroyed." << std::endl;
}

RTCGeometry EmbreeGeometry::handle() const
{
    return m_handle;
}

void EmbreeGeometry::setTransform(const Transform& T)
{
    m_T = T;
}

void EmbreeGeometry::setTransform(const Matrix4x4& T)
{
    // scale?
    Transform T2;
    T2.set(T);
    setTransform(T2);
}

void setTransformAndScale(const Matrix4x4& T)
{

}

Transform EmbreeGeometry::transform() const
{
    return m_T;
}

void EmbreeGeometry::setScale(const Vector3& S)
{
    m_S = S;
}

Vector3 EmbreeGeometry::scale() const
{
    return m_S;
}

void EmbreeGeometry::disable()
{
    rtcDisableGeometry(m_handle);
}

void EmbreeGeometry::enable()
{
    rtcEnableGeometry(m_handle);
}

void EmbreeGeometry::release()
{
    rtcReleaseGeometry(m_handle);
}

void EmbreeGeometry::commit()
{
    rtcCommitGeometry(m_handle);
}



} // namespace rmagine

