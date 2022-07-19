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
    handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_INSTANCE);
}

EmbreeInstance::~EmbreeInstance()
{
    
}

unsigned int EmbreeInstance::id() const
{
    return instID;
}

void EmbreeInstance::setScene(EmbreeScenePtr scene)
{
    m_scene = scene;
    instID = rtcAttachGeometry(scene->handle(), handle);
    rtcReleaseGeometry(handle);
}

EmbreeScenePtr EmbreeInstance::scene()
{
    return m_scene;
}

void EmbreeInstance::setMesh(EmbreeMeshPtr mesh)
{
    m_mesh = mesh;
    rtcSetGeometryInstancedScene(handle, mesh->scene()->handle() );
}

EmbreeMeshPtr EmbreeInstance::mesh()
{
    return m_mesh;
}

void EmbreeInstance::commit()
{
    rtcSetGeometryTransform(handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &T.data[0][0]);
    rtcCommitGeometry(handle);
}

} // namespace rmagine