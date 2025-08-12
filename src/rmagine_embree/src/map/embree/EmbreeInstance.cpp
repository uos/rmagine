#include "rmagine/map/embree/EmbreeInstance.hpp"

// other internal deps
#include "rmagine/map/embree/EmbreeDevice.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"

#include <embree4/rtcore.h>

#include <iostream>

#include <map>
#include <cassert>

namespace rmagine 
{


// TODO: implement instance level point query support
// - See: https://github.com/RenderKit/embree/blob/master/tutorials/closest_point/closest_point_device.cpp)

EmbreeInstance::EmbreeInstance(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_INSTANCE);
    // std::cout << "[EmbreeInstance::EmbreeInstance()] constructed." << std::endl;
}

EmbreeInstance::~EmbreeInstance()
{
    // std::cout << "[EmbreeInstance::~EmbreeInstance()] destroyed." << std::endl;
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
    Matrix4x4 M;
    M.set(m_T);

    Matrix4x4 Ms;
    Ms.setIdentity();
    Ms(0,0) = m_S.x;
    Ms(1,1) = m_S.y;
    Ms(2,2) = m_S.z;

    M = M * Ms;

    rtcSetGeometryTransform(m_handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &M(0,0));
}

} // namespace rmagine