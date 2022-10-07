#include "rmagine/map/embree/EmbreeScene.hpp"

#include <rmagine/map/embree/EmbreeInstance.hpp>
#include <rmagine/map/embree/EmbreeMesh.hpp>

#include <iostream>

#include <map>
#include <cassert>

#include <rmagine/util/prints.h>
#include <rmagine/math/assimp_conversions.h>
#include <rmagine/util/assimp/helper.h>

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
    
    size_t nparents_before = geom->parents.size();
    geom->parents.insert(weak_from_this());
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

EmbreeInstancePtr EmbreeScene::instantiate()
{
    EmbreeInstancePtr geom_inst = std::make_shared<EmbreeInstance>(m_device);
    geom_inst->set(shared_from_this());
    return geom_inst;
}

std::unordered_set<EmbreeGeometryPtr> EmbreeScene::findLeafs() const
{
    std::unordered_set<EmbreeGeometryPtr> ret;

    for(auto elem : m_geometries)
    {
        EmbreeInstancePtr inst = std::dynamic_pointer_cast<EmbreeInstance>(elem.second);
        if(inst)
        {
            // is instance
            EmbreeScenePtr inst_scene = inst->scene();
            std::unordered_set<EmbreeGeometryPtr> ret2 = inst_scene->findLeafs();
            // integrate ret2 in ret
            ret.insert(ret2.begin(), ret2.end());
        } else {
            ret.insert(elem.second);
        }
    }

    return ret;
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

void EmbreeScene::freeze()
{
    // std::cout << "[EmbreeScene::freeze()] start optimizing scene.." << std::endl;
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
            // std::cout << "- instance " << instance_id << " optimized to mesh " << geom_id << std::endl;
        }
    }

    // std::cout << "[EmbreeScene::freeze()] finished optimizing scene.." << std::endl;
}

EmbreeScenePtr make_embree_scene(
    const aiScene* ascene,
    EmbreeDevicePtr device)
{   
    EmbreeSceneSettings settings = {};
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>(settings, device);

    // std::vector<EmbreeMeshPtr> meshes;

    std::map<unsigned int, EmbreeMeshPtr> meshes;

    // 1. meshes
    // std::cout << "[make_embree_scene()] Loading Meshes..." << std::endl;
    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        // std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
        const aiMesh* amesh = ascene->mMeshes[i];

        if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
        {
            // triangle mesh
            EmbreeMeshPtr mesh = std::make_shared<EmbreeMesh>(amesh);
            mesh->commit();
            meshes[i] = mesh;
        } else {
            std::cout << "[ make_embree_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
        }
    }

    std::unordered_set<EmbreeGeometryPtr> instanciated_meshes;

    // 2. instances
    const aiNode* root_node = ascene->mRootNode;
    std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);

    // std::cout << "[make_embree_scene()] Loading Instances..." << std::endl;
    for(size_t i=0; i<mesh_nodes.size(); i++)
    {
        const aiNode* node = mesh_nodes[i];
        
        Matrix4x4 M = global_transform(node);
        Transform T;
        Vector3 scale;
        decompose(M, T, scale);

        EmbreeScenePtr mesh_scene = std::make_shared<EmbreeScene>();

        for(unsigned int i = 0; i<node->mNumMeshes; i++)
        {
            unsigned int mesh_id = node->mMeshes[i];
            auto mesh_it = meshes.find(mesh_id);
            if(mesh_it != meshes.end())
            {
                // mesh found
                EmbreeMeshPtr mesh = mesh_it->second;
                instanciated_meshes.insert(mesh);
                mesh_scene->add(mesh);
                mesh_scene->commit();
            } else {
                std::cout << "[make_embree_scene()] WARNING: could not find mesh_id " 
                    << mesh_id << " in meshes during instantiation" << std::endl;
            }
        }

        mesh_scene->commit();

        // std::cout << "--- mesh added to mesh_scene" << std::endl;
        EmbreeInstancePtr mesh_instance = std::make_shared<EmbreeInstance>();
        mesh_instance->set(mesh_scene);
        mesh_instance->name = node->mName.C_Str();
        mesh_instance->setTransform(T);
        mesh_instance->setScale(scale);
        mesh_instance->apply();
        mesh_instance->commit();
        // std::cout << "--- mesh_instance created" << std::endl;
        scene->add(mesh_instance);
    }

    // std::cout << "add meshes that are not instanciated ..." << std::endl;
    // ADD MESHES THAT ARE NOT INSTANCIATED
    for(auto elem : meshes)
    {
        auto mesh = elem.second;
        if(instanciated_meshes.find(mesh) == instanciated_meshes.end())
        {
            // mesh was never instanciated. add to scene
            scene->add(mesh);
        }
    }

    // std::cout << "[make_embree_scene()] Scene loaded." << std::endl;

    return scene;
}

} // namespace rmagine