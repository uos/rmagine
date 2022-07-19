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

unsigned int EmbreeScene::add(EmbreeInstancePtr inst)
{
    unsigned int inst_id = rtcAttachGeometry(m_scene, inst->handle());
    m_instances[inst_id] = inst;
    inst->parent = shared_from_this();
    inst->release();
    return inst_id;
}

std::unordered_map<unsigned int, EmbreeInstancePtr> EmbreeScene::instances() const
{
    return m_instances;
}

bool EmbreeScene::hasInstance(unsigned int inst_id) const
{
    return m_instances.find(inst_id) != m_instances.end();
}

EmbreeInstancePtr EmbreeScene::removeInstance(unsigned int inst_id)
{
    EmbreeInstancePtr ret;

    if(m_instances.find(inst_id) != m_instances.end())
    {
        ret = m_instances[inst_id];
        m_instances.erase(inst_id);
    }

    return ret;
}

unsigned int EmbreeScene::add(EmbreeMeshPtr mesh)
{
    unsigned int geom_id = rtcAttachGeometry(m_scene, mesh->handle());
    m_meshes[geom_id] = mesh;
    mesh->parent = shared_from_this();
    mesh->release();
    return geom_id;
}

std::unordered_map<unsigned int, EmbreeMeshPtr> EmbreeScene::meshes() const
{
    return m_meshes;
}

bool EmbreeScene::hasMesh(unsigned int mesh_id) const
{
    return m_meshes.find(mesh_id) != m_meshes.end();
}

EmbreeMeshPtr EmbreeScene::removeMesh(unsigned int mesh_id)
{
    EmbreeMeshPtr ret;

    if(m_meshes.find(mesh_id) != m_meshes.end())
    {
        ret = m_meshes[mesh_id];
        m_meshes.erase(mesh_id);
    }

    return ret;
}

RTCScene EmbreeScene::handle()
{
    return m_scene;
}

void EmbreeScene::commit()
{
    rtcCommitScene(m_scene);
}

void EmbreeScene::optimize()
{
    std::unordered_map<unsigned int, unsigned int> inst_mesh_map;

    for(auto elem : m_instances)
    {
        unsigned int inst_id = elem.first;
        EmbreeInstancePtr inst = elem.second;

        if(inst->scene()->parents.size() == 1 && inst->scene()->meshes().size() == 1)
        {
            EmbreeMeshPtr mesh = inst->scene()->meshes().begin()->second;
            mesh->setTransform(inst->T);

            inst->T.setIdentity();
            add(mesh);
            mesh->commit();
            std::cout << elem.first <<  " Can be optimized!" << std::endl;
        }
    }

    for(auto it = m_instances.begin(); it != m_instances.end(); )
    {
        EmbreeInstancePtr inst = it->second;
        if(inst->scene()->parents.size() == 1 && inst->scene()->meshes().size() == 1)
        {
            inst->disable();
            it = m_instances.erase(it);
        } else {
            ++it;
        }
    }

    commit();
}

} // namespace rmagine