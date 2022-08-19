#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/map/optix/OptixGeometry.hpp"
#include "rmagine/map/optix/OptixMesh.hpp"
#include "rmagine/map/optix/OptixInst.hpp"
#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

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

    for(auto elem : m_geometries)
    {
        elem.second->cleanupParents();
    }
}

unsigned int OptixScene::add(OptixGeometryPtr geom)
{
    auto it = m_ids.find(geom);
    if(it == m_ids.end())
    {
        if(m_geometries.empty())
        {
            // first element
            m_geom_type = geom->type();
            if(m_geom_type == OptixGeometryType::INSTANCE)
            {
                m_type = OptixSceneType::INSTANCES;
            } else {
                m_type = OptixSceneType::GEOMETRIES;
            }
        }

        if(geom->type() != m_geom_type)
        {
            std::cout << "[OptixScene::add()] WARNING - used mixed types of geometries. NOT supported" << std::endl;
        }

        // add geom to self
        unsigned int id = gen.get();
        m_geometries[id] = geom;
        m_ids[geom] = id;

        // add self to geom
        geom->addParent(this_shared<OptixScene>());

        return id;
    } else {
        return it->second;
    }
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
    if(m_type == OptixSceneType::INSTANCES)
    {
        buildIAS();
    } else if(m_type == OptixSceneType::GEOMETRIES ) {
        buildGAS();
    }

}

unsigned int OptixScene::depth() const
{
    unsigned int ret = 0;

    for(auto elem : m_geometries)
    {
        ret = std::max(ret, elem.second->depth());
    }

    return ret + 1;
}

void OptixScene::cleanupParents()
{
    for(auto it = m_parents.cbegin(); it != m_parents.cend();)
    {
        if (it->expired())
        {
            m_parents.erase(it++);    // or "it = m.erase(it)" since C++11
        } else {
            ++it;
        }
    }
}


std::unordered_set<OptixInstPtr> OptixScene::parents() const
{
    std::unordered_set<OptixInstPtr> ret;

    for(OptixInstWPtr elem : m_parents)
    {
        if(auto tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void OptixScene::addParent(OptixInstPtr parent)
{
    m_parents.insert(parent);
}


void OptixScene::buildGAS()
{
    std::cout << "SCENE BUILD GAS" << std::endl;
    size_t n_build_inputs = m_geometries.size();

    OptixBuildInput build_inputs[n_build_inputs];
    
    size_t idx = 0;
    for(auto elem : m_geometries)
    {
        OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(elem.second);
        if(mesh)
        {
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = mesh->vertices.size();
            triangle_input.triangleArray.vertexBuffers = mesh->getVertexBuffer();
        
            // FACES
            triangle_input.triangleArray.indexFormat  = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = mesh->faces.size();
            triangle_input.triangleArray.indexBuffer         = mesh->getFaceBuffer();

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };
            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            build_inputs[idx] = triangle_input;
        } else {
            std::cout << "WARNING COULD NOT FILL GAS INPUTS" << std::endl;
        }

        idx ++;
    }


    // Acceleration Options
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};

    unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;

    { // BUILD FLAGS
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }

    accel_options.buildFlags = build_flags;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                m_ctx->ref(),
                &accel_options,
                build_inputs,
                n_build_inputs, // Number of build inputs
                &gas_buffer_sizes
                ) );

    
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes) );
    
    if(!m_as)
    {
        m_as = std::make_shared<OptixAccelerationStructure>();
    }

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );
    m_as->buffer_size = gas_buffer_sizes.outputSizeInBytes;
    m_as->n_elements = n_build_inputs;

    OPTIX_CHECK( optixAccelBuild(
                m_ctx->ref(),
                m_stream->handle(),                  // CUDA stream
                &accel_options,
                build_inputs,
                n_build_inputs,                  // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                m_as->buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &m_as->handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
}

void OptixScene::buildIAS()
{

    // fill m_hitgroup_data

    // { // meshes
    //     unsigned int n_meshes = m_geometries.rbegin()->first + 1;
    //     if(!m_h_hitgroup_data.size())
    //     {
    //         m_h_hitgroup_data.resize(1);
    //     }

    //     Memory<MeshAttributes, RAM> attr_cpu(n_meshes);
    //     for(auto elem : m_geometries)
    //     {
    //         OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(elem.second);
    //         if(mesh)
    //         {
    //             attr_cpu[elem.first].face_normals = mesh->face_normals.raw();
    //             attr_cpu[elem.first].vertex_normals = mesh->vertex_normals.raw();
    //         } else {
    //             std::cout << "NO MESH: how to handle normals?" << std::endl;
    //         }
    //     }

    //     if(m_h_hitgroup_data[0].n_meshes != n_meshes)
    //     {
    //         // Number of meshes changed! Recreate
    //         if(m_h_hitgroup_data[0].mesh_attributes)
    //         {
    //             cudaFree(m_h_hitgroup_data[0].mesh_attributes);
    //         }

    //         // create space for mesh attributes
    //         cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].mesh_attributes), 
    //             n_meshes * sizeof(MeshAttributes));

    //         // std::cout << "HITGROUP DATA: Created space for " << n_meshes << " meshes" << std::endl;
    //     }

    //     m_h_hitgroup_data[0].n_meshes = n_meshes;

    //     // copy mesh attributes
    //     CUDA_CHECK( cudaMemcpy(
    //                 reinterpret_cast<void*>(m_h_hitgroup_data[0].mesh_attributes),
    //                 reinterpret_cast<void*>(attr_cpu.raw()),
    //                 attr_cpu.size() * sizeof(MeshAttributes),
    //                 cudaMemcpyHostToDevice
    //                 ) );

    //     // std::cout << "HITGROUP DATA: Copied " << n_meshes << " meshes" << std::endl;
    // } // meshes

    // { // connections inst -> mesh + instances
    //     Memory<unsigned int, RAM> inst_to_mesh;

    //     OptixInstancesPtr insts = std::dynamic_pointer_cast<OptixInstances>(m_root);

    //     if(insts)
    //     {
    //         auto instances = insts->instances();
    //         size_t Ninstances = instances.rbegin()->first + 1;

    //         inst_to_mesh.resize(Ninstances);
    //         for(unsigned int i=0; i<inst_to_mesh.size(); i++)
    //         {
    //             inst_to_mesh[i] = -1;
    //         }

    //         for(auto elem : instances)
    //         {
    //             unsigned int inst_id = elem.first;
    //             OptixGeometryPtr geom = elem.second->geometry();
    //             OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(geom);

    //             if(mesh)
    //             {
    //                 unsigned int mesh_id = get(mesh);
    //                 inst_to_mesh[inst_id] = mesh_id;
    //             }
    //         }
    //     } else {
    //         // only one mesh 0 -> 0
    //         inst_to_mesh.resize(1);
    //         inst_to_mesh[0] = 0;
    //     }

    //     unsigned int n_instances = inst_to_mesh.size();

    //     if(m_h_hitgroup_data[0].n_instances != n_instances)
    //     {
    //         // Number of instances changed! Recreate
    //         if(m_h_hitgroup_data[0].inst_to_mesh)
    //         {
    //             cudaFree(m_h_hitgroup_data[0].inst_to_mesh);
    //         }

    //         if(m_h_hitgroup_data[0].instances_attributes)
    //         {
    //             cudaFree(m_h_hitgroup_data[0].instances_attributes);
    //         }

    //         // create space for mesh attributes
    //         cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].inst_to_mesh), 
    //             n_instances * sizeof(int));

    //         // create space for mesh attributes
    //         cudaMalloc(reinterpret_cast<void**>(&m_h_hitgroup_data[0].instances_attributes), 
    //             n_instances * sizeof(InstanceAttributes));
        
    //         // std::cout << "HITGROUP DATA: Created space for " << n_instances << " instances" << std::endl;
    //     }
    //     m_h_hitgroup_data[0].n_instances = n_instances;

    //     CUDA_CHECK( cudaMemcpy(
    //                 reinterpret_cast<void*>(m_h_hitgroup_data[0].inst_to_mesh),
    //                 reinterpret_cast<void*>(inst_to_mesh.raw()),
    //                 inst_to_mesh.size() * sizeof(unsigned int),
    //                 cudaMemcpyHostToDevice
    //                 ) );

    //     // std::cout << "HITGROUP DATA: Copied " << n_instances << " instances" << std::endl;
    // }

    // // std::cout << "UPLOAD HITGROUP DATA" << std::endl;
    // // m_hitgroup_data = m_h_hitgroup_data;
}

OptixScenePtr make_optix_scene(
    const aiScene* ascene, 
    OptixContextPtr context)
{
    OptixScenePtr scene = std::make_shared<OptixScene>(context);

    // // 1. meshes
    // // std::cout << "[make_optix_scene()] Loading Meshes..." << std::endl;

    // for(size_t i=0; i<ascene->mNumMeshes; i++)
    // {
    //     // std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
    //     const aiMesh* amesh = ascene->mMeshes[i];

    //     if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
    //     {
    //         // triangle mesh
    //         OptixMeshPtr mesh = std::make_shared<OptixMesh>(amesh, context);
    //         mesh->commit();
    //         scene->add(mesh);
    //     } else {
    //         std::cout << "[ make_optix_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
    //     }
    // }

    // // tmp
    // auto meshes = scene->geometries();

    // // 2. instances (if available)
    // std::unordered_set<OptixGeometryPtr> instanciated_meshes;

    // const aiNode* root_node = ascene->mRootNode;
    // std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);
    
    // // std::cout << "[make_optix_scene()] Loading Instances..." << std::endl;

    // OptixInstancesPtr insts = std::make_shared<OptixInstances>(context);

    // for(size_t i=0; i<mesh_nodes.size(); i++)
    // {
    //     const aiNode* node = mesh_nodes[i];
    //     // std::cout << "[make_optix_scene()] - " << i << ": " << node->mName.C_Str();
    //     // std::cout << ", total path: ";

    //     // std::vector<std::string> path = path_names(node);   
    //     // for(auto name : path)
    //     // {
    //     //     std::cout << name << "/";
    //     // }
    //     // std::cout << std::endl;

    //     Matrix4x4 M = global_transform(node);
    //     Transform T;
    //     Vector3 scale;
    //     decompose(M, T, scale);

    //     std::vector<OptixInstPtr> mesh_insts;
        
        
    //     if(node->mNumMeshes > 1)
    //     {
    //         // std::cout << "[make_optix_scene()] Optix Warning: More than one mesh per instance? TODO make this possible" << std::endl;
        
    //         // make flat hierarchy: one instance per mesh
    //         for(unsigned int i = 0; i < node->mNumMeshes; i++)
    //         {
    //             unsigned int mesh_id = node->mMeshes[i];
    //             auto mesh_it = meshes.find(mesh_id);
    //             if(mesh_it != meshes.end())
    //             {
    //                 OptixGeometryPtr mesh = mesh_it->second;
    //                 // mark as instanciated
    //                 instanciated_meshes.insert(mesh);

    //                 OptixInstPtr mesh_inst = std::make_shared<OptixInst>(context);
    //                 mesh_inst->setGeometry(mesh);
    //                 mesh_inst->name = std::string(node->mName.C_Str()) + "/" + mesh->name;
    //                 mesh_insts.push_back(mesh_inst);
    //             } else {
    //                 // TODO: warning
    //                 std::cout << "[make_optix_scene()] WARNING could not find mesh_id " << mesh_id << " in meshes" << std::endl;
    //             }
    //         }
    //     } else {
    //         unsigned int mesh_id = node->mMeshes[0];
    //         auto mesh_it = meshes.find(mesh_id);
    //         if(mesh_it != meshes.end())
    //         {
    //             OptixGeometryPtr mesh = mesh_it->second;
    //             // mark as instanciated
    //             instanciated_meshes.insert(mesh);

    //             OptixInstPtr mesh_inst = std::make_shared<OptixInst>(context);
    //             mesh_inst->setGeometry(mesh);
    //             mesh_inst->name = node->mName.C_Str();
    //             mesh_insts.push_back(mesh_inst);
    //         } else {
    //             // TODO: warning
    //             std::cout << "[make_optix_scene()] WARNING could not find mesh_id " 
    //                 << mesh_id << " in meshes during instantiation" << std::endl;
    //         }
    //     }


    //     for(auto mesh_inst : mesh_insts)
    //     {
    //         mesh_inst->setTransform(T);
    //         mesh_inst->setScale(scale);
    //         mesh_inst->apply();
    //         insts->add(mesh_inst);
    //     }
    // }

    // if(instanciated_meshes.size() == 0)
    // {
    //     // SINGLE GAS
    //     if(scene->geometries().size() == 1)
    //     {
    //         scene->setRoot(scene->geometries().begin()->second);
    //     } else {
    //         std::cout << "[make_optix_scene()] " << scene->geometries().size() << " unconnected meshes!" << std::endl;
    //     }
    // } else {
        
    //     if(instanciated_meshes.size() != meshes.size())
    //     {
    //         std::cout << "[make_optix_scene()] There are some meshes left" << std::endl;
    //     }
    //     insts->commit();
    //     scene->setRoot(insts);
    // }

    return scene;
}




} // namespace rmagine