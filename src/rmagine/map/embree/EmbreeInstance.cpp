#include "rmagine/map/embree/EmbreeInstance.hpp"

#include <iostream>

#include <map>
#include <cassert>

#include <rmagine/map/EmbreeMesh.hpp>
#include <rmagine/map/EmbreeScene.hpp>


namespace rmagine {

EmbreeInstance::EmbreeInstance(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_INSTANCE);
    std::cout << "[EmbreeInstance::EmbreeInstance()] constructed." << std::endl;
}

EmbreeInstance::~EmbreeInstance()
{
    std::cout << "[EmbreeInstance::~EmbreeInstance()] destroyed." << std::endl;
}

void EmbreeInstance::setTransform(const Matrix4x4& T)
{
    this->T = T;
}

void EmbreeInstance::setTransform(const Transform& T)
{
    this->T.set(T);
}

void EmbreeInstance::set(EmbreeScenePtr scene)
{
    m_scene = scene;
    rtcSetGeometryInstancedScene(m_handle, m_scene->handle());
    // shared_from_this
    m_scene->parents.insert(std::dynamic_pointer_cast<EmbreeInstance>(shared_from_this()));
}

EmbreeScenePtr EmbreeInstance::scene()
{
    return m_scene;
}

void EmbreeInstance::apply()
{
    rtcSetGeometryTransform(m_handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &T.data[0][0]);
}

} // namespace rmagine