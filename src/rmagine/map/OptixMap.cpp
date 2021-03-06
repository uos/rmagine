#include "rmagine/map/OptixMap.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/GenericAlign.hpp"

#include <optix.h>
#include <optix_stubs.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>

#include <map>

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

namespace rmagine {

OptixMap::OptixMap(const aiScene* ascene, int device)
:m_device(device)
{
    initContext(device);

    fillMeshes(ascene);

    if(meshes.size() > 1)
    {
        // enable instance level
        m_instance_level = true;
    } else if(meshes.size() == 1) {
        m_instance_level = false;
    } else {
        throw std::runtime_error("No Meshes?");
    }

    // Build instance acceleration structure IAS if required
    if(meshes.size() > 1)
    {
        // Build GASes
        for(unsigned int mesh_id = 0; mesh_id < meshes.size(); mesh_id++)
        {
            // OptixMesh& mesh = meshes[mesh_id];
            buildGAS(meshes[mesh_id], meshes[mesh_id].gas);
        }

        // Get Instance information and fill the instances memory
        fillInstances(ascene);
        // Build the GAS
        buildIAS(instances, as);

    } else {
        buildGAS(meshes[0], as);
    }
}

OptixMap::~OptixMap()
{
    // std::cout << "Destruct OptixMap" << std::endl;
    cudaFree( reinterpret_cast<void*>( as.buffer ) );
    optixDeviceContextDestroy( context );

    // std::cout << "Free " << instances.size() << " instances" << std::endl;
    // std::cout << "Free " << meshes.size() << " meshes" << std::endl;

    if(meshes.size() > 1)
    {
        for(size_t i=0; i<meshes.size(); i++)
        {
            cudaFree( reinterpret_cast<void*>( meshes[i].gas.buffer ) );
        }
    }
}

void OptixMap::initContext(int device)
{
    int driver;
    cudaDriverGetVersion(&driver);
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);

    std::cout << cuda_version << std::endl;

    std::stringstream driver_version_str, cuda_version_str;
    driver_version_str << driver / 1000 << "." << (driver % 1000) / 10 << "." << driver % 10;
    cuda_version_str << cuda_version / 1000 << "." << (cuda_version % 1000) / 10 << "." << cuda_version % 10;

    std::cout << "[OptixMesh] Latest CUDA for driver: " << driver_version_str.str() << ". Current CUDA version: " << cuda_version_str.str() << std::endl;

    // Initialize CUDA
    cudaDeviceProp info;
    CUDA_CHECK( cudaGetDeviceProperties(&info, device) );

    
    cuCtxCreate(&cuda_context, 0, device);

    // Check flags
    cuInit(0);

    std::cout << "[OptixMesh] Initialized CUDA context on device " << device << ": " << info.name << " " << info.luid << std::endl;

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK( optixInit() );

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 3;

    OPTIX_CHECK( optixDeviceContextCreate( cuda_context, &options, &context ) );

    std::stringstream optix_version_str;
    optix_version_str << OPTIX_VERSION / 10000 << "." << (OPTIX_VERSION % 10000) / 100 << "." << OPTIX_VERSION % 100;

    std::cout << "[OptixMesh] Init Optix (" << optix_version_str.str() << ") context on latest CUDA context " << std::endl;

}

void OptixMap::fillMeshes(const aiScene* ascene)
{
    meshes.resize(ascene->mNumMeshes);
    for(size_t mesh_id=0; mesh_id < ascene->mNumMeshes; mesh_id++)
    {
        const aiMesh* ai_mesh = ascene->mMeshes[mesh_id];
        const aiVector3D* ai_vertices = ai_mesh->mVertices;
        unsigned int num_vertices = ai_mesh->mNumVertices;
        const aiFace* ai_faces = ai_mesh->mFaces;
        unsigned int num_faces = ai_mesh->mNumFaces;

        // OptixInstance instance;
        OptixMesh     mesh;

        mesh.vertices.resize(num_vertices);
        mesh.faces.resize(num_faces);
        mesh.normals.resize(num_faces);

        Memory<Point, RAM> vertices_cpu(num_vertices);
        Memory<Face, RAM> faces_cpu(num_faces);
        Memory<Vector, RAM> normals_cpu(num_faces);

        // convert
        for(size_t i=0; i<num_vertices; i++)
        {
            vertices_cpu[i] = {
                    ai_vertices[i].x,
                    ai_vertices[i].y,
                    ai_vertices[i].z};
        }

        for(size_t i=0; i<num_faces; i++)
        {
            faces_cpu[i].v0 = ai_faces[i].mIndices[0];
            faces_cpu[i].v1 = ai_faces[i].mIndices[1];
            faces_cpu[i].v2 = ai_faces[i].mIndices[2];
        }
        
        for(size_t i=0; i<num_faces; i++)
        {
            unsigned int v0_id = ai_faces[i].mIndices[0];
            unsigned int v1_id = ai_faces[i].mIndices[1];
            unsigned int v2_id = ai_faces[i].mIndices[2];

            const Vector v0{ai_vertices[v0_id].x, ai_vertices[v0_id].y, ai_vertices[v0_id].z};
            const Vector v1{ai_vertices[v1_id].x, ai_vertices[v1_id].y, ai_vertices[v1_id].z};
            const Vector v2{ai_vertices[v2_id].x, ai_vertices[v2_id].y, ai_vertices[v2_id].z};
            
            Vector n = (v1-v0).normalized().cross((v2 - v0).normalized());
            n.normalize();

            normals_cpu[i] = {
                n.x,
                n.y,
                n.z};
        }

        mesh.vertices = vertices_cpu;
        mesh.faces = faces_cpu;
        mesh.normals = normals_cpu;
        meshes[mesh_id] = mesh;
    }
}

void OptixMap::fillInstances(const aiScene* ascene)
{
    // get Transform from aiscene
    
    std::map<unsigned int, aiMatrix4x4> Tmap;

    // Parsing transformation tree
    unsigned int geom_id = 0;
    const aiNode* root_node = ascene->mRootNode;
    for(unsigned int i=0; i<root_node->mNumChildren; i++)
    {
        const aiNode* n = root_node->mChildren[i];
        if(n->mNumChildren == 0)
        {
            // Leaf
            if(n->mNumMeshes > 0)
            {
                aiMatrix4x4 aT = n->mTransformation;
                Tmap[n->mMeshes[0]] = aT;
            }
        } else {
            // TODO: handle deeper tree. concatenate transformations
            // std::cout << "- Children: " << n->mNumChildren << std::endl;
        }
    }

    if(Tmap.size() > 0)
    {
        instances.resize(ascene->mNumMeshes);
        for(size_t mesh_id=0; mesh_id < ascene->mNumMeshes; mesh_id++)
        {
            auto fit = Tmap.find(mesh_id);
            bool Tfound = false;
            aiMatrix4x4 T;
            if(fit != Tmap.end())
            {
                // Found transform!
                T = fit->second;
                Tfound = true;
            }
            
            // convert 
            OptixInstance& instance = instances[mesh_id];

            // Build IAS
            instance.transform[ 0] = T.a1; // Rxx
            instance.transform[ 1] = T.a2; // Rxy
            instance.transform[ 2] = T.a3; // Rxz
            instance.transform[ 3] = T.a4; // tx
            instance.transform[ 4] = T.b1; // Ryx
            instance.transform[ 5] = T.b2; // Ryy
            instance.transform[ 6] = T.b3; // Ryz
            instance.transform[ 7] = T.b4; // ty 
            instance.transform[ 8] = T.c1; // Rzx
            instance.transform[ 9] = T.c2; // Rzy
            instance.transform[10] = T.c3; // Rzz
            instance.transform[11] = T.c4; // tz
            // ..
            instance.instanceId = mesh_id;
            instance.sbtOffset = 0;
            instance.visibilityMask = 255;
            // you could override the geometry flags here: OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = meshes[mesh_id].gas.handle;
        }
    }

    std::cout << "Finished filling Instances" << std::endl;
}

void OptixMap::buildGAS(
    const OptixMesh& mesh, 
    OptixAccelerationStructure& gas)
{
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput triangle_input = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // m_vertices = reinterpret_cast<CUdeviceptr>(mesh.vertices.raw());
    CUdeviceptr tmp = reinterpret_cast<CUdeviceptr>(mesh.vertices.raw());

    // VERTICES
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
    triangle_input.triangleArray.numVertices   = mesh.vertices.size();
    triangle_input.triangleArray.vertexBuffers = &tmp;

    // FACES
    // std::cout << "- define faces" << std::endl;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
    triangle_input.triangleArray.numIndexTriplets    = mesh.faces.size();
    triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(mesh.faces.raw());

    // ADDITIONAL SETTINGS
    triangle_input.triangleArray.flags         = triangle_input_flags;
    // TODO: this is bad. I define the sbt records inside the programs. 
    triangle_input.triangleArray.numSbtRecords = 1;

    // Acceleration Options
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
#ifndef NDEBUG
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
#else
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
#endif
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                context,
                &accel_options,
                &triangle_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ) );

    // std::cout << "OptiX GAS requires " << static_cast<double>(gas_buffer_sizes.outputSizeInBytes) / 1000000.0 << " MB of Memory" << std::endl;

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &gas.buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                context,
                0,                  // CUDA stream
                &accel_options,
                &triangle_input,
                1,                  // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                gas.buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &gas.handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );
    
    // TODO: Compact

    // // We can now free the scratch space buffer used during build and the vertex
    // // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
}

void OptixMap::buildIAS(
    const Memory<OptixInstance, VRAM_CUDA>& instances,
    OptixAccelerationStructure& ias)
{
    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.numInstances = instances.size();
    instance_input.instanceArray.instances = reinterpret_cast<CUdeviceptr>(instances.raw());

    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( 
        context, 
        &ias_accel_options, 
        &instance_input, 
        1, 
        &ias_buffer_sizes ) );

    CUdeviceptr d_temp_buffer_ias;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_ias ),
        ias_buffer_sizes.tempSizeInBytes) );

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &ias.buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild( 
        context, 
        0, 
        &ias_accel_options, 
        &instance_input, 
        1, 
        d_temp_buffer_ias,
        ias_buffer_sizes.tempSizeInBytes, 
        ias.buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &ias.handle,
        nullptr, 
        0 
        ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );
}

void OptixMap::buildIAS(
    const Memory<OptixInstance, RAM>& instances,
    OptixAccelerationStructure& ias)
{
    // copy to gpu
    Memory<OptixInstance, VRAM_CUDA> instances_gpu;
    instances_gpu = instances;
    buildIAS(instances_gpu, ias);
}

} // namespace mamcl