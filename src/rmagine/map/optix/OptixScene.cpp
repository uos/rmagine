#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/map/optix/OptixGeometry.hpp"
#include "rmagine/map/optix/OptixMesh.hpp"
#include "rmagine/map/optix/OptixInstances.hpp"
#include "rmagine/map/optix/OptixInst.hpp"

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>

#include <optix_stubs.h>

#include <rmagine/math/assimp_conversions.h>
#include <rmagine/util/assimp/helper.h>
#include <rmagine/math/linalg.h>


namespace rmagine
{

OptixScene::OptixScene(OptixContextPtr context)
:OptixEntity(context)
{
    
}

OptixScene::OptixScene(OptixGeometryPtr geom, OptixContextPtr context)
:OptixEntity(context)
,m_geom(geom)
{
    
}

OptixScene::~OptixScene()
{

}

void OptixScene::setRoot(OptixGeometryPtr geom)
{
    m_geom = geom;
}

OptixGeometryPtr OptixScene::getRoot() const
{
    return m_geom;
}

unsigned int OptixScene::add(OptixGeometryPtr geom)
{
    unsigned int id = gen.get();
    m_geometries[id] = geom;
    m_ids[geom] = id;
    return id;
}

unsigned int OptixScene::get(OptixGeometryPtr geom) const
{
    return m_ids.at(geom);
}

std::map<unsigned int, OptixGeometryPtr> OptixScene::geometries() const
{
    return m_geometries;
}

std::unordered_map<OptixGeometryPtr, unsigned int> OptixScene::ids() const
{
    return m_ids;
}


OptixScenePtr make_optix_scene(const aiScene* ascene)
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    // 1. meshes
    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        // std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
        const aiMesh* amesh = ascene->mMeshes[i];

        if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
        {
            // triangle mesh
            OptixMeshPtr mesh = std::make_shared<OptixMesh>(amesh);
            mesh->commit();
            scene->add(mesh);
        } else {
            std::cout << "[ make_embree_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
        }
    }

    // tmp
    auto meshes = scene->geometries();

    // 2. instances (if available)
    std::unordered_set<OptixGeometryPtr> instanciated_meshes;

    const aiNode* root_node = ascene->mRootNode;
    std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);
    
    std::cout << "[make_embree_scene()] Loading Instances..." << std::endl;

    OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    for(size_t i=0; i<mesh_nodes.size(); i++)
    {
        const aiNode* node = mesh_nodes[i];
        std::cout << "- " << i << ": " << node->mName.C_Str();
        std::cout << ", total path: ";

        std::vector<std::string> path = path_names(node);   
        for(auto name : path)
        {
            std::cout << name << "/";
        }
        std::cout << std::endl;

        Matrix4x4 M = global_transform(node);
        Transform T;
        Vector3 scale;
        decompose(M, T, scale);

        OptixInstPtr mesh_inst = std::make_shared<OptixInst>();
        
        if(node->mNumMeshes > 1)
        {
            std::cout << "Optix Warning: More than one mesh per instance? TODO make this possible" << std::endl; 
        } else {
            unsigned int mesh_id = node->mMeshes[0];
            auto mesh_it = meshes.find(mesh_id);
            if(mesh_it != meshes.end())
            {
                OptixGeometryPtr mesh = mesh_it->second;
                instanciated_meshes.insert(mesh);
                mesh_inst->setGeometry(mesh);
            }
        }

        mesh_inst->name = node->mName.C_Str();
        mesh_inst->setTransform(T);
        mesh_inst->setScale(scale);
        mesh_inst->apply();

        insts->add(mesh_inst);
    }

    if(instanciated_meshes.size() == 0)
    {
        // SINGLE GAS
        if(scene->geometries().size() == 1)
        {
            scene->setRoot(scene->geometries().begin()->second);
        } else {
            std::cout << scene->geometries().size() << " unconnected meshes!" << std::endl;
        }
    } else {
        
        if(instanciated_meshes.size() != meshes.size())
        {
            std::cout << "There are some meshes left" << std::endl;
        }
        insts->commit();
        scene->setRoot(insts);
    }

    return scene;
}


} // namespace rmagine