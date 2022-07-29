
#include "rmagine/map/embree/EmbreeGeometry.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"

#include <iostream>
#include <Eigen/Dense>

namespace rmagine
{

EmbreeGeometry::EmbreeGeometry(EmbreeDevicePtr device)
:m_device(device)
,m_S{1.0,1.0,1.0}
{
    m_T.setIdentity();
    // std::cout << "[EmbreeGeometry::EmbreeGeometry()] constructed." << std::endl;
}

EmbreeGeometry::~EmbreeGeometry()
{
    release();
    // std::cout << "[EmbreeGeometry::~EmbreeGeometry()] destroyed." << std::endl;
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

Matrix4x4 EmbreeGeometry::matrix() const
{
    return compose(m_T, m_S);
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

void EmbreeGeometry::cleanup_parents()
{
    for(auto it = parents.begin(); it != parents.end();)
    {
        if(it->lock())
        {
            ++it;
        } else {
            it = parents.erase(it);
        }
    }
}

std::unordered_map<unsigned int, EmbreeSceneWPtr> EmbreeGeometry::ids()
{
    std::unordered_map<unsigned int, EmbreeSceneWPtr> ret;

    cleanup_parents();

    for(auto it = parents.begin(); it != parents.end(); ++it)
    {
        if(auto parent = it->lock())
        {
            // parent exists
            ret[parent->get(shared_from_this())] = parent;
        }
    }

    return ret;
}



} // namespace rmagine

