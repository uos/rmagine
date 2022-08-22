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
    if(sbt_data.geometries)
    {
        cudaFree(sbt_data.geometries);
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

        m_geom_added = true;

        return id;
    } else {
        return it->second;
    }
}

unsigned int OptixScene::get(OptixGeometryPtr geom) const
{
    return m_ids.at(geom);
}

std::optional<unsigned int> OptixScene::getOpt(OptixGeometryPtr geom) const
{
    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        return it->second;
    }

    return {};
}

bool OptixScene::has(OptixGeometryPtr geom) const
{
    return (m_ids.find(geom) != m_ids.end());
}

bool OptixScene::has(unsigned int geom_id) const
{
    return (m_geometries.find(geom_id) != m_geometries.end());
}

bool OptixScene::remove(OptixGeometryPtr geom)
{
    bool ret = false;

    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        unsigned int geom_id = it->second;

        m_ids.erase(it);
        m_geometries.erase(geom_id);
        geom->removeParent(this_shared<OptixScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;
        ret = true;
    }

    return ret;
}

OptixGeometryPtr OptixScene::remove(unsigned int geom_id)
{
    OptixGeometryPtr ret;

    auto it = m_geometries.find(geom_id);
    if(it != m_geometries.end())
    {
        OptixGeometryPtr geom = it->second;

        m_geometries.erase(it);
        m_ids.erase(geom);
        geom->removeParent(this_shared<OptixScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;

        ret = geom;
    }

    return ret;
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

OptixInstPtr OptixScene::instantiate()
{
    OptixInstPtr ret = std::make_shared<OptixInst>();
    ret->set(this_shared<OptixScene>());
    return ret;
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
    // std::cout << "SCENE BUILD GAS" << std::endl;

    size_t n_build_inputs = m_geometries.size();

    OptixBuildInput build_inputs[n_build_inputs];

    OptixSceneSBT sbt_data_h;
    cudaMallocHost(&sbt_data_h.geometries, sizeof(OptixGeomSBT) * n_build_inputs);

    // TODO make proper realloc
    sbt_data.n_geometries = n_build_inputs;
    sbt_data.type = m_type;
    CUDA_CHECK( cudaMalloc(&sbt_data.geometries, sizeof(OptixGeomSBT) * n_build_inputs) );

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

            if(mesh->pre_transform)
            {
                triangle_input.triangleArray.transformFormat =  OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
                triangle_input.triangleArray.preTransform        = mesh->pre_transform;
            }
            

            // ADDITIONAL SETTINGS
            // move them to mesh object
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };

            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            build_inputs[idx] = triangle_input;

            // SBT data
            sbt_data_h.geometries[idx].mesh_data = mesh->sbt_data;
            sbt_data_h.geometries[idx].mesh_data.id = elem.first;
            // std::cout << "Connect GAS SBT " << idx << " -> Mesh " << elem.first << std::endl;
        } else {
            std::cout << "WARNING COULD NOT FILL GAS INPUTS" << std::endl;
        }

        idx ++;
    }

    // copy sbt
    CUDA_CHECK( cudaMemcpyAsync(
        sbt_data.geometries, 
        sbt_data_h.geometries, 
        sizeof(OptixGeomSBT) * n_build_inputs, 
        cudaMemcpyHostToDevice, m_stream->handle()) );

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

    if(!m_geom_added && !m_geom_removed && m_as)
    {
        // UPDATE
        // std::cout << "GAS - UPDATE!" << std::endl;
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        // BUILD   
        // std::cout << "GAS - BUILD!" << std::endl;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    m_geom_added = false;
    m_geom_removed = false;


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
        // make new
        m_as = std::make_shared<OptixAccelerationStructure>();
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );
    } else {
        if(m_as->buffer_size != gas_buffer_sizes.outputSizeInBytes)
        {
            // realloc
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_as->buffer ),
                    gas_buffer_sizes.outputSizeInBytes
                    ) );
        }
    }
    
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

    // std::cout << "GAS constructed" << std::endl;

}

void OptixScene::buildIAS()
{
    // std::cout << "[OptixScene::buildIAS()] start." << std::endl;
    const size_t n_instances = m_geometries.size();

    // fill m_hitgroup_data
    Memory<OptixInstance, RAM> inst_h(n_instances);

    OptixSceneSBT sbt_data_h;
    cudaMallocHost(&sbt_data_h.geometries, sizeof(OptixGeomSBT) * n_instances);

    size_t idx = 0;
    for(auto elem : m_geometries)
    {
        unsigned int inst_id = elem.first;
        OptixInstPtr inst = std::dynamic_pointer_cast<OptixInst>(elem.second);
        inst_h[idx] = inst->data();
        inst_h[idx].instanceId = elem.first;

        sbt_data_h.geometries[idx].inst_data = inst->sbt_data;

        idx++;
    }

    // std::cout << "- COPY INSTANCE DATA" << std::endl;
    
    // COPY INSTANCES DATA
    CUdeviceptr m_inst_buffer;
    CUDA_CHECK( cudaMalloc( 
        reinterpret_cast<void**>( &m_inst_buffer ), 
        inst_h.size() * sizeof(OptixInstance) ) );

    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( m_inst_buffer ),
                inst_h.raw(),
                inst_h.size() * sizeof(OptixInstance),
                cudaMemcpyHostToDevice,
                m_stream->handle()
                ) );

    

    // std::cout << "- COPY SBT DATA" << std::endl;

    

    if(n_instances > sbt_data.n_geometries)
    {
        CUDA_CHECK( cudaFree( sbt_data.geometries ) );
        CUDA_CHECK( cudaMalloc(&sbt_data.geometries, sizeof(OptixGeomSBT) * n_instances) );
    }

    sbt_data.n_geometries = n_instances;
    sbt_data.type = m_type;

    // COPY INSTANCES SBT DATA
    CUDA_CHECK( cudaMemcpyAsync(
        sbt_data.geometries,
        sbt_data_h.geometries,
        sizeof(OptixGeomSBT) * n_instances,
        cudaMemcpyHostToDevice,
        m_stream->handle()
    ) );

    // we dont need the host memory anymore
    cudaFreeHost(sbt_data_h.geometries);


    // std::cout << "- MAKE BUILD INPUT" << std::endl;
    // BEGIN WITH BUILD INPUT

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.numInstances = inst_h.size();
    instance_input.instanceArray.instances = m_inst_buffer;

    OptixAccelBuildOptions ias_accel_options = {};
    unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;
    { // BUILD FLAGS
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        #if OPTIX_VERSION >= 73000
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
        #endif
    }

    ias_accel_options.buildFlags = build_flags;
    ias_accel_options.motionOptions.numKeys = 1;
    
    if(!m_geom_added && !m_geom_removed && m_as)
    {
        // UPDATE
        // std::cout << "IAS - UPDATE!" << std::endl;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        // BUILD   
        // std::cout << "IAS - BUILD!" << std::endl;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    m_geom_added = false;
    m_geom_removed = false;



    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( 
        m_ctx->ref(), 
        &ias_accel_options,
        &instance_input, 
        1, 
        &ias_buffer_sizes ) );

    
    CUdeviceptr d_temp_buffer_ias;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_ias ),
        ias_buffer_sizes.tempSizeInBytes) );

    
    if(!m_as)
    {
        // make new
        m_as = std::make_shared<OptixAccelerationStructure>();
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );
    } else {
        if(m_as->buffer_size != ias_buffer_sizes.outputSizeInBytes)
        {
            // realloc
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_as->buffer ),
                    ias_buffer_sizes.outputSizeInBytes
                    ) );
        }
    }
    
    m_as->buffer_size = ias_buffer_sizes.outputSizeInBytes;
    m_as->n_elements = n_instances;


    OPTIX_CHECK(optixAccelBuild( 
        m_ctx->ref(), 
        m_stream->handle(), 
        &ias_accel_options, 
        &instance_input, 
        1, // num build inputs
        d_temp_buffer_ias,
        ias_buffer_sizes.tempSizeInBytes, 
        m_as->buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &m_as->handle,
        nullptr, 
        0 
    ));

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );

    // std::cout << "[OptixScene::buildIAS()] done." << std::endl;
}



OptixScenePtr make_optix_scene(
    const aiScene* ascene, 
    OptixContextPtr context)
{
    OptixScenePtr scene = std::make_shared<OptixScene>(context);

    std::map<unsigned int, OptixMeshPtr> meshes;
    // 1. meshes
    // std::cout << "[make_optix_scene()] Loading Meshes..." << std::endl;

    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        // std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
        const aiMesh* amesh = ascene->mMeshes[i];

        if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
        {
            // triangle mesh
            OptixMeshPtr mesh = std::make_shared<OptixMesh>(amesh);
            mesh->commit();
            meshes[i] = mesh;
            // std::cout << "Mesh " << i << "(" << mesh->name << ") added." << std::endl;
        } else {
            std::cout << "[ make_optix_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
        }
    }

    // 2. instances (if available)
    std::unordered_set<OptixGeometryPtr> instanciated_meshes;
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

        OptixScenePtr mesh_scene = std::make_shared<OptixScene>();

        for(unsigned int i = 0; i<node->mNumMeshes; i++)
        {
            unsigned int mesh_id = node->mMeshes[i];
            auto mesh_it = meshes.find(mesh_id);
            if(mesh_it != meshes.end())
            {
                // mesh found
                OptixMeshPtr mesh = mesh_it->second;
                instanciated_meshes.insert(mesh);
                mesh_scene->add(mesh);
                mesh_scene->commit();
            } else {
                std::cout << "[make_optix_scene()] WARNING: could not find mesh_id " 
                    << mesh_id << " in meshes during instantiation" << std::endl;
            }
        }

        mesh_scene->commit();

        // std::cout << "--- mesh added to mesh_scene" << std::endl;
        OptixInstPtr mesh_instance = std::make_shared<OptixInst>();
        mesh_instance->set(mesh_scene);
        mesh_instance->name = node->mName.C_Str();
        mesh_instance->setTransform(T);
        mesh_instance->setScale(scale);
        mesh_instance->apply();
        mesh_instance->commit();
        // std::cout << "--- mesh_instance created" << std::endl;
        unsigned int inst_id = scene->add(mesh_instance);
        // std::cout << "Instance " << inst_id << " (" << mesh_instance->name << ") added" << std::endl;
    }

    // if(scene->type() != OptixSceneType::INSTANCES)
    // {
    //     std::cout << "add meshes that are not instanciated as geometry" << std::endl;
    // } else {
    //     std::cout << "add meshes that are not instanciated as instance" << std::endl;
    // }

    // ADD MESHES THAT ARE NOT INSTANCIATED
    for(auto elem : meshes)
    {
        auto mesh = elem.second;
        if(instanciated_meshes.find(mesh) == instanciated_meshes.end())
        {
            // mesh was never instanciated. add to scene
            if(scene->type() != OptixSceneType::INSTANCES)
            {
                scene->add(mesh);
            } else {
                // mesh->instantiate();
                OptixInstPtr geom_inst = mesh->instantiate();
                geom_inst->apply();
                geom_inst->commit();
                scene->add(geom_inst);
            }
        }
    }

    return scene;
}




} // namespace rmagine