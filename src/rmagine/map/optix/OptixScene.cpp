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

OptixScene::OptixScene(OptixGeometryPtr root, OptixContextPtr context)
:OptixEntity(context)
,m_root(root)
{
    
}

OptixScene::~OptixScene()
{
    if(m_h_hitgroup_data.size())
    {
        if(m_h_hitgroup_data[0].mesh_attributes)
        {
            cudaFree(m_h_hitgroup_data[0].mesh_attributes);
        }

        if(m_h_hitgroup_data[0].inst_to_mesh)
        {
            cudaFree(m_h_hitgroup_data[0].inst_to_mesh);
        }

        if(m_h_hitgroup_data[0].instances_attributes)
        {
            cudaFree(m_h_hitgroup_data[0].instances_attributes);
        }
    }
}

void OptixScene::setRoot(OptixGeometryPtr root)
{
    m_root = root;
}

OptixGeometryPtr OptixScene::getRoot() const
{
    return m_root;
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

void OptixScene::commit()
{
    // fill m_hitgroup_data

    { // meshes
        unsigned int n_meshes = m_geometries.rbegin()->first + 1;
        if(!m_h_hitgroup_data.size())
        {
            m_h_hitgroup_data.resize(1);
        }

        Memory<MeshAttributes, RAM> attr_cpu(n_meshes);
        for(auto elem : m_geometries)
        {
            OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(elem.second);
            if(mesh)
            {
                attr_cpu[elem.first].face_normals = mesh->face_normals.raw();
                attr_cpu[elem.first].vertex_normals = mesh->vertex_normals.raw();
            } else {
                std::cout << "NO MESH: how to handle normals?" << std::endl;
            }
        }

        if(m_h_hitgroup_data[0].n_meshes != n_meshes)
        {
            // Number of meshes changed! Recreate
            if(m_h_hitgroup_data[0].mesh_attributes)
            {
                cudaFree(m_h_hitgroup_data[0].mesh_attributes);
            }

            // create space for mesh attributes
            cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].mesh_attributes), 
                n_meshes * sizeof(MeshAttributes));

            // std::cout << "HITGROUP DATA: Created space for " << n_meshes << " meshes" << std::endl;
        }

        m_h_hitgroup_data[0].n_meshes = n_meshes;

        // copy mesh attributes
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(m_h_hitgroup_data[0].mesh_attributes),
                    reinterpret_cast<void*>(attr_cpu.raw()),
                    attr_cpu.size() * sizeof(MeshAttributes),
                    cudaMemcpyHostToDevice
                    ) );

        // std::cout << "HITGROUP DATA: Copied " << n_meshes << " meshes" << std::endl;
    } // meshes

    { // connections inst -> mesh + instances
        Memory<unsigned int, RAM> inst_to_mesh;

        OptixInstancesPtr insts = std::dynamic_pointer_cast<OptixInstances>(m_root);

        if(insts)
        {
            auto instances = insts->instances();
            size_t Ninstances = instances.rbegin()->first + 1;

            inst_to_mesh.resize(Ninstances);
            for(unsigned int i=0; i<inst_to_mesh.size(); i++)
            {
                inst_to_mesh[i] = -1;
            }

            for(auto elem : instances)
            {
                unsigned int inst_id = elem.first;
                OptixGeometryPtr geom = elem.second->geometry();
                OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(geom);

                if(mesh)
                {
                    unsigned int mesh_id = get(mesh);
                    inst_to_mesh[inst_id] = mesh_id;
                }
            }
        } else {
            // only one mesh 0 -> 0
            inst_to_mesh.resize(1);
            inst_to_mesh[0] = 0;
        }

        unsigned int n_instances = inst_to_mesh.size();

        if(m_h_hitgroup_data[0].n_instances != n_instances)
        {
            // Number of instances changed! Recreate
            if(m_h_hitgroup_data[0].inst_to_mesh)
            {
                cudaFree(m_h_hitgroup_data[0].inst_to_mesh);
            }

            if(m_h_hitgroup_data[0].instances_attributes)
            {
                cudaFree(m_h_hitgroup_data[0].instances_attributes);
            }

            // create space for mesh attributes
            cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].inst_to_mesh), 
                n_instances * sizeof(int));

            // create space for mesh attributes
            cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].instances_attributes), 
                n_instances * sizeof(InstanceAttributes));
        
            // std::cout << "HITGROUP DATA: Created space for " << n_instances << " instances" << std::endl;
        }
        m_h_hitgroup_data[0].n_instances = n_instances;

        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(m_h_hitgroup_data[0].inst_to_mesh),
                    reinterpret_cast<void*>(inst_to_mesh.raw()),
                    inst_to_mesh.size() * sizeof(unsigned int),
                    cudaMemcpyHostToDevice
                    ) );

        // std::cout << "HITGROUP DATA: Copied " << n_instances << " instances" << std::endl;
    }

    // std::cout << "UPLOAD HITGROUP DATA" << std::endl;
    // m_hitgroup_data = m_h_hitgroup_data;
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