#include "imagine/map/OptixMap.hpp"
#include "imagine/util/optix/OptixDebug.hpp"
#include "imagine/types/MemoryCuda.hpp"


#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include "imagine/util/GenericAlign.hpp"

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

namespace imagine {

OptixMap::OptixMap(const aiScene* ascene, int device)
:m_device(device)
{
    initContext(device);

    if(ascene->mNumMeshes > 1)
    {
        std::cout << "OptixMesh WARNING: multiple meshes found in scene. TODO: implement" << std::endl;
    }

    size_t mesh_id = 0;

    const aiMesh* mesh = ascene->mMeshes[mesh_id];
    const aiVector3D* ai_vertices = mesh->mVertices;
    m_num_vertices = mesh->mNumVertices;
    
    const aiFace* ai_faces = mesh->mFaces;
    m_num_faces = mesh->mNumFaces;

    // std::cout << "Upload " << m_num_vertices << " Vertices and " 
    //     << m_num_faces << " Faces" << std::endl;

    // std::cout << "Additional Mesh Attributes: " << std::endl;
    // std::cout << "- face normals" << std::endl;

    normals.resize(mesh->mNumFaces);

    // Init buffers
    Memory<float3, RAM> vertices_cpu(m_num_vertices);
    Memory<uint32_t, RAM> faces_cpu(m_num_faces * 3);
    Memory<float3, RAM> normals_cpu(normals.size());

    for(size_t i=0; i<m_num_vertices; i++)
    {
        vertices_cpu[i] = make_float3(
                ai_vertices[i].x,
                ai_vertices[i].y,
                ai_vertices[i].z);
    }

    for(size_t i=0; i<m_num_faces; i++)
    {
        faces_cpu[i*3+0] = ai_faces[i].mIndices[0];
        faces_cpu[i*3+1] = ai_faces[i].mIndices[1];
        faces_cpu[i*3+2] = ai_faces[i].mIndices[2];
    }
    
    for(size_t i=0; i<mesh->mNumFaces; i++)
    {
        unsigned int v0_id = ai_faces[i].mIndices[0];
        unsigned int v1_id = ai_faces[i].mIndices[1];
        unsigned int v2_id = ai_faces[i].mIndices[2];

        const Eigen::Vector3d v0(ai_vertices[v0_id].x, ai_vertices[v0_id].y, ai_vertices[v0_id].z);
        const Eigen::Vector3d v1(ai_vertices[v1_id].x, ai_vertices[v1_id].y, ai_vertices[v1_id].z);
        const Eigen::Vector3d v2(ai_vertices[v2_id].x, ai_vertices[v2_id].y, ai_vertices[v2_id].z);
        
        Eigen::Vector3d n = ( (v1-v0).normalized() ).cross((v2-v0).normalized());
        n.normalize();

        normals_cpu[i] = make_float3(
            n.x(),
            n.y(),
            n.z());
    }

    normals = normals_cpu;

    const size_t vertices_bytes = sizeof(float3) * m_num_vertices;
    const size_t faces_bytes = sizeof(uint32_t) * 3 * m_num_faces;

    CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &m_vertices ), 
                vertices_bytes
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( m_vertices ),
                vertices_cpu.raw(),
                vertices_bytes,
                cudaMemcpyHostToDevice
                ) );
    
    CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &m_faces ), 
                faces_bytes 
                ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( m_faces ),
                faces_cpu.raw(),
                faces_bytes,
                cudaMemcpyHostToDevice
                ) );

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput triangle_input = {};
    triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // VERTICES
    triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices   = m_num_vertices;
    triangle_input.triangleArray.vertexBuffers = &m_vertices;

    // FACES
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes  = 3 * sizeof(uint32_t);
    triangle_input.triangleArray.numIndexTriplets    = m_num_faces;
    triangle_input.triangleArray.indexBuffer         = m_faces;

    // ADDITIONAL SETTINGS
    triangle_input.triangleArray.flags         = triangle_input_flags;
    // TODO: this is bad. I define the sbt records inside the programs. 
    triangle_input.triangleArray.numSbtRecords = 1;

    // Acceleration Options
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
#ifdef DEBUG
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
                reinterpret_cast<void**>( &d_gas_output_buffer ),
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
                d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &gas_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );
    
    // TODO: Compact

    // // We can now free the scratch space buffer used during build and the vertex
    // // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
}

OptixMap::~OptixMap()
{
    cudaFree( reinterpret_cast<void*>( m_vertices ) );
    cudaFree( reinterpret_cast<void*>( m_faces ) );

    cudaFree( reinterpret_cast<void*>( d_gas_output_buffer ) );

    OPTIX_CHECK( optixDeviceContextDestroy( context ) );
}

void OptixMap::initContext(int device)
{
    // Initialize CUDA
    cudaDeviceProp info;
    CUDA_CHECK( cudaGetDeviceProperties(&info, device) );
    std::cout << "[OptixMesh] Init context on device " << device << " " << info.name << " " << info.luid << std::endl;

    cuCtxCreate(&cuda_context, 0, device);

    // Check flags
    cuInit(0);

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK( optixInit() );

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 3;

    OPTIX_CHECK( optixDeviceContextCreate( cuda_context, &options, &context ) );
}

} // namespace mamcl