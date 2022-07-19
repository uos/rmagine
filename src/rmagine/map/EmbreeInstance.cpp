#include "rmagine/map/EmbreeInstance.hpp"

#include <iostream>

#include <map>
#include <cassert>

#include <rmagine/map/EmbreeMesh.hpp>
#include <rmagine/map/EmbreeScene.hpp>


namespace rmagine {

// EmbreeInstancePtr instance(new EmbreeInstance());

EmbreeInstance::EmbreeInstance(EmbreeDevicePtr device)
{
    m_device = device;
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_INSTANCE);
}

EmbreeInstance::~EmbreeInstance()
{
    std::cout << "Destroy Instance" << std::endl;
    rtcDisableGeometry(m_handle);

}

void EmbreeInstance::setTransform(const Matrix4x4& T)
{
    this->T = T;
}

void EmbreeInstance::setTransform(const Transform& T)
{
    this->T.set(T);
}

RTCGeometry EmbreeInstance::handle() const
{
    return m_handle;
}

void EmbreeInstance::set(EmbreeScenePtr scene)
{
    m_scene = scene;
    rtcSetGeometryInstancedScene(m_handle, m_scene->handle());
    m_scene->parents.insert(shared_from_this());
}

EmbreeScenePtr EmbreeInstance::scene()
{
    return m_scene;
}

void EmbreeInstance::commit()
{
    rtcSetGeometryTransform(m_handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &T.data[0][0]);
    rtcCommitGeometry(m_handle);
}

void EmbreeInstance::disable()
{
    rtcDisableGeometry(m_handle);
}

void EmbreeInstance::release()
{
    rtcReleaseGeometry(m_handle);
}

} // namespace rmagine