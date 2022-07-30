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
}

EmbreeScene::~EmbreeScene()
{
    // std::cout << "[EmbreeScene::~EmbreeScene()] start destroying." << std::endl;

    for(auto elem : m_geometries)
    {
        rtcDetachGeometry(m_scene, elem.first);
        elem.second->cleanupParents();
    }

    m_geometries.clear();
    m_ids.clear();

    rtcReleaseScene(m_scene);
    // std::cout << "[EmbreeScene::~EmbreeScene()] destroyed." << std::endl;
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
    m_ids[geom] = geom_id;
    // TODO: geometry can be attached to multiple scenes!
    size_t nparents_before = geom->parents.size();
    // std::cout << "parents before: " << geom->parents.size() << std::endl;
    geom->parents.insert(weak_from_this());
    // std::cout << "parents after: " << geom->parents.size() << std::endl;
    size_t nparents_after = geom->parents.size();


    if(nparents_after == nparents_before)
    {
        std::cout << "WARNING geometry seems to be already added before. same number of parents as before: " << nparents_after << std::endl; 
    }
    
    // geom->id = geom_id;
    return geom_id;
}

std::optional<unsigned int> EmbreeScene::getOpt(
    const EmbreeGeometryPtr geom) const
{
    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        return it->second;
    }
    return {};
}

std::optional<unsigned int> EmbreeScene::getOpt(
    const std::shared_ptr<const EmbreeGeometry> geom) const
{
    return getOpt(std::const_pointer_cast<EmbreeGeometry>(geom));
}

unsigned int EmbreeScene::get(const std::shared_ptr<EmbreeGeometry> geom) const
{
    return m_ids.at(geom);
}

unsigned int EmbreeScene::get(const std::shared_ptr<const EmbreeGeometry> geom) const
{
    // TODO check
    return get(std::const_pointer_cast<EmbreeGeometry>(geom));
}

bool EmbreeScene::has(const EmbreeGeometryPtr geom) const
{
    return m_ids.find(geom) != m_ids.end();
}

bool EmbreeScene::has(const std::shared_ptr<const EmbreeGeometry> geom) const
{
    return has(std::const_pointer_cast<EmbreeGeometry>(geom));
}

bool EmbreeScene::remove(EmbreeGeometryPtr geom)
{
    bool ret = false;

    auto geom_id_opt = getOpt(geom);
    if(geom_id_opt)
    {
        unsigned int geom_id = *geom_id_opt;
        rtcDetachGeometry(m_scene, geom_id);
        size_t nelements = geom->parents.erase(shared_from_this());
        if(nelements == 0)
        {
            std::cout << "WARNING could not remove self from childs parents" << std::endl;
        }
        
        m_geometries.erase(geom_id);
        m_ids.erase(geom);
        ret = true;
    }

    return ret;
}

EmbreeGeometryPtr EmbreeScene::get(const unsigned int geom_id) const
{
    EmbreeGeometryPtr ret;
    auto it = m_geometries.find(geom_id);
    if(it != m_geometries.end())
    {
        ret = it->second;
    }
    return ret;
}

bool EmbreeScene::has(const unsigned int geom_id) const
{
    return m_geometries.find(geom_id) != m_geometries.end();
}

EmbreeGeometryPtr EmbreeScene::remove(const unsigned int geom_id)
{
    EmbreeGeometryPtr geom;

    if(has(geom_id))
    {
        geom = m_geometries[geom_id];

        rtcDetachGeometry(m_scene, geom_id);
        size_t nelements = geom->parents.erase(shared_from_this());
        if(nelements == 0)
        {
            std::cout << "WARNING could not remove self from childs parents" << std::endl;
        }
        
        m_geometries.erase(geom_id);
        m_ids.erase(geom);
    }

    return geom;
}

std::unordered_map<EmbreeGeometryPtr, unsigned int> EmbreeScene::ids() const
{
    return m_ids;
}

std::unordered_map<unsigned int, EmbreeGeometryPtr> EmbreeScene::geometries() const
{
    return m_geometries;
}

RTCScene EmbreeScene::handle()
{
    return m_scene;
}

void EmbreeScene::commit()
{
    rtcCommitScene(m_scene);
    m_committed_once = true;
}

bool EmbreeScene::committedOnce() const
{
    return m_committed_once;
}

bool EmbreeScene::isTopLevel() const
{
    return parents.empty();
}

std::unordered_map<unsigned int, unsigned int> EmbreeScene::integrate(EmbreeScenePtr other)
{
    std::unordered_map<unsigned int, unsigned int> integration_map;
    for(auto elem : other->geometries())
    {
        unsigned int old_id = elem.first;
        unsigned int new_id = add(elem.second);
        std::cout << "integrating " << old_id << " -> " << new_id << std::endl;
        integration_map[old_id] = new_id;
    }

    return integration_map;
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
        unsigned int instance_id = m_ids[instance];
        remove(instance_id);

        EmbreeMeshPtr mesh = std::dynamic_pointer_cast<EmbreeMesh>(
            instance->scene()->geometries().begin()->second);

        if(mesh)
        {
            // TODO check if this is correct
            mesh->setScale(instance->scale().mult_ewise(mesh->scale()));

            mesh->setTransform(instance->transform() * mesh->transform());
            mesh->apply();
            mesh->commit();

            unsigned int geom_id = add(mesh);
            std::cout << "- instance " << instance_id << " optimized to mesh " << geom_id << std::endl;
        }
    }

    std::cout << "[EmbreeScene::optimize()] finished optimizing scene.." << std::endl;
}

} // namespace rmagine