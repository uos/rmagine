#include "rmagine/map/embree/EmbreeScene.hpp"

#include <rmagine/map/embree/EmbreeInstance.hpp>
#include <rmagine/map/embree/EmbreeMesh.hpp>

#include <iostream>

#include <map>
#include <cassert>

namespace rmagine {


EmbreeScene::EmbreeScene(
    EmbreeSceneSettings settings, 
    EmbreeDevicePtr device)
:m_device(device)
,m_scene(rtcNewScene(device->handle()))
{
    setQuality(settings.quality);
    setFlags(settings.flags);
    std::cout << "[EmbreeScene::EmbreeScene()] constructed." << std::endl;
}

EmbreeScene::~EmbreeScene()
{
    std::cout << "[EmbreeScene::~EmbreeScene()] start destroying." << std::endl;

    for(auto elem : m_geometries)
    {
        rtcDetachGeometry(m_scene, elem.first);
        // reset self. should be automatically done by weak ptr
        // elem.second->parent.reset();
    }

    m_geometries.clear();
    std::cout << "[EmbreeScene::~EmbreeScene()] release scene." << std::endl;
    rtcReleaseScene(m_scene);
    std::cout << "[EmbreeScene::~EmbreeScene()] destroyed." << std::endl;
}

void EmbreeScene::setQuality(RTCBuildQuality quality)
{
    rtcSetSceneBuildQuality(m_scene, quality);
}

void EmbreeScene::setFlags(RTCSceneFlags flags)
{
    rtcSetSceneFlags(m_scene, flags);
}

unsigned int EmbreeScene::add(EmbreeGeometryPtr geom)
{
    unsigned int geom_id = rtcAttachGeometry(m_scene, geom->handle());
    m_geometries[geom_id] = geom;
    geom->parent = weak_from_this();
    geom->id = geom_id;
    return geom_id;
}

std::unordered_map<unsigned int, EmbreeGeometryPtr> EmbreeScene::geometries() const
{
    return m_geometries;
}

bool EmbreeScene::has(unsigned int geom_id) const
{
    return m_geometries.find(geom_id) != m_geometries.end();
}

EmbreeGeometryPtr EmbreeScene::remove(unsigned int geom_id)
{
    EmbreeGeometryPtr ret;

    if(has(geom_id))
    {
        rtcDetachGeometry(m_scene, geom_id);
        ret = m_geometries[geom_id];
        ret->parent.reset();
        m_geometries.erase(geom_id);
    }

    return ret;
}

// template<typename T>
// unsigned int EmbreeScene::count() const
// {
//     unsigned int ret = 0;

//     for(auto it = m_geometries.begin(); it != m_geometries.end(); )
//     {
//         if(std::dynamic_pointer_cast<T>(it->second))
//         {
//             ret++;
//         }
//     }

//     return ret;
// }

// std::unordered_map<unsigned int, EmbreeInstancePtr> EmbreeScene::instances() const
// {
//     return m_instances;
// }

// bool EmbreeScene::hasInstance(unsigned int inst_id) const
// {
//     return m_instances.find(inst_id) != m_instances.end();
// }

// EmbreeInstancePtr EmbreeScene::removeInstance(unsigned int inst_id)
// {
//     EmbreeInstancePtr ret;

//     if(m_instances.find(inst_id) != m_instances.end())
//     {
//         rtcDetachGeometry(m_scene, inst_id);
//         ret = m_instances[inst_id];
//         ret->parent.reset();
//         m_instances.erase(inst_id);
//     }

//     return ret;
// }

// unsigned int EmbreeScene::add(EmbreeMeshPtr mesh)
// {
//     unsigned int geom_id = rtcAttachGeometry(m_scene, mesh->handle());
//     m_meshes[geom_id] = mesh;
//     mesh->parent = weak_from_this();
//     mesh->id = geom_id;
//     mesh->release();
//     return geom_id;
// }

// std::unordered_map<unsigned int, EmbreeMeshPtr> EmbreeScene::meshes() const
// {
//     return m_meshes;
// }

// bool EmbreeScene::hasMesh(unsigned int mesh_id) const
// {
//     return m_meshes.find(mesh_id) != m_meshes.end();
// }

// EmbreeMeshPtr EmbreeScene::removeMesh(unsigned int mesh_id)
// {
//     EmbreeMeshPtr ret;

//     if(m_meshes.find(mesh_id) != m_meshes.end())
//     {
//         rtcDetachGeometry(m_scene, mesh_id);
//         ret = m_meshes[mesh_id];
//         ret->parent.reset();
//         m_meshes.erase(mesh_id);
//     }

//     return ret;
// }

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
    std::cout << "[EmbreeScene::optimize()] start optimizing scene.." << std::endl;


    std::vector<EmbreeInstancePtr> instances_to_optimize;

    for(auto it = m_geometries.begin(); it != m_geometries.end(); ++it)
    {
        EmbreeInstancePtr instance = std::dynamic_pointer_cast<EmbreeInstance>(it->second);

        if(instance)
        {
            if(instance->scene()->parents.size() == 1 && instance->scene()->geometries().size() == 1)
            {
                EmbreeMeshPtr mesh = std::dynamic_pointer_cast<EmbreeMesh>(
                    instance->scene()->geometries().begin()->second);
                if(mesh)
                {
                    instances_to_optimize.push_back(instance);
                }
            }
        }
    }

    // remove all
    for(auto instance : instances_to_optimize)
    {
        remove(instance->id);
        if(instance->parent.lock())
        {
            std::cout << "WARNING " << instance->id << " was not removed correctly" << std::endl;
        }

        EmbreeMeshPtr mesh = std::dynamic_pointer_cast<EmbreeMesh>(
            instance->scene()->geometries().begin()->second);

        if(mesh)
        {

            // Matrix4x4 T;
            // T.set(mesh->transform());
            // Matrix4x4 S;
            // S.setIdentity();
            // S(0,0) = mesh->scale().x;
            // S(1,1) = mesh->scale().y;
            // S(2,2) = mesh->scale().z;

            // total transform: first scale than isometry
            // T = T * S;

            // TODO: elemeninate Matrix4x4
            mesh->setTransform(instance->T);
            mesh->apply();
            mesh->commit();

            // instance->T.setIdentity();
            unsigned int geom_id = add(mesh);
            std::cout << "- instance " << instance->id << " optimized to mesh " << geom_id << std::endl;
        }
    }

    std::cout << "[EmbreeScene::optimize()] finished optimizing scene.." << std::endl;
    // commit();
}

} // namespace rmagine