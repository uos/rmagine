#include "rmagine/map/EmbreeScene.hpp"

#include <rmagine/map/EmbreeInstance.hpp>
#include <rmagine/map/EmbreeMesh.hpp>

#include <iostream>

#include <map>
#include <cassert>

namespace rmagine {


EmbreeScene::EmbreeScene(EmbreeDevicePtr device, EmbreeSceneSettings settings)
{
    m_scene = rtcNewScene(device->handle());
    setQuality(settings.quality);
    setFlags(settings.flags);
}

EmbreeScene::~EmbreeScene()
{
    m_instances.clear();
    m_meshes.clear();
    rtcReleaseScene(m_scene);
}

void EmbreeScene::setQuality(RTCBuildQuality quality)
{
    rtcSetSceneBuildQuality(m_scene, quality);
}

void EmbreeScene::setFlags(RTCSceneFlags flags)
{
    rtcSetSceneFlags(m_scene, flags);
}

void EmbreeScene::add(EmbreeInstancePtr inst)
{
    m_instances.insert(inst);
}

EmbreeInstanceSet EmbreeScene::instances() const
{
    return m_instances;
}

void EmbreeScene::remove(EmbreeInstancePtr inst)
{
    size_t nerased = m_instances.erase(inst);
    if(nerased > 0)
    {
        EmbreeScenePtr none;
        inst->setScene(none);
    }
}

void EmbreeScene::add(EmbreeMeshPtr mesh)
{
    m_meshes.insert(mesh);
}

void EmbreeScene::remove(EmbreeMeshPtr mesh)
{
    size_t nerased = m_meshes.erase(mesh);
    if(nerased > 0)
    {
        EmbreeScenePtr none;
        mesh->setScene(none);
    }
}



RTCScene EmbreeScene::handle()
{
    return m_scene;
}

void EmbreeScene::commit()
{
    rtcCommitScene(m_scene);
}

} // namespace rmagine