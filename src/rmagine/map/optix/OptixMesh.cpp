#include "rmagine/map/optix/OptixMesh.hpp"

#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include "rmagine/math/assimp_conversions.h"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/GenericAlign.hpp"
#include "rmagine/map/mesh_preprocessing.cuh"
#include "rmagine/math/math.cuh"

#include "rmagine/util/cuda/CudaStream.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <rmagine/util/prints.h>

namespace rmagine
{

OptixMesh::OptixMesh(OptixContextPtr context)
:Base(context)
{
    // std::cout << "[OptixMesh::OptixMesh()] constructed." << std::endl;
}

OptixMesh::OptixMesh(
    const aiMesh* amesh, 
    OptixContextPtr context)
:OptixMesh(context)
{
    const aiVector3D* avertices = amesh->mVertices;
    unsigned int num_vertices = amesh->mNumVertices;
    const aiFace* afaces = amesh->mFaces;
    unsigned int num_faces = amesh->mNumFaces;

    name = amesh->mName.C_Str();

    vertices.resize(num_vertices);
    faces.resize(num_faces);

    Memory<Point, RAM> vertices_cpu(num_vertices);
    Memory<Face, RAM> faces_cpu(num_faces);

    // convert
    for(size_t i=0; i<num_vertices; i++)
    {
        vertices_cpu[i] = {
                avertices[i].x,
                avertices[i].y,
                avertices[i].z};
    }

    for(size_t i=0; i<num_faces; i++)
    {
        faces_cpu[i].v0 = afaces[i].mIndices[0];
        faces_cpu[i].v1 = afaces[i].mIndices[1];
        faces_cpu[i].v2 = afaces[i].mIndices[2];
    }

    vertices = vertices_cpu;
    faces = faces_cpu;
    computeFaceNormals();
    apply();
}

OptixMesh::~OptixMesh()
{
    // std::cout << "[OptixMesh::~OptixMesh()] destroyed." << std::endl;
}

void OptixMesh::apply()
{
    // TODO
    Memory<Matrix4x4, RAM> M(1);
    M[0] = matrix();

    Memory<Matrix4x4, VRAM_CUDA> M_;
    M_ = M;

    if(vertices_.size() != vertices.size())
    {
        vertices_.resize(vertices.size());
    }
    
    mult1xN(M_, vertices, vertices_);

    m_vertices_ref = reinterpret_cast<CUdeviceptr>(vertices_.raw());



    // if(face_normals.size() > 0)
    // {
    //     if(face_normals_.size() != face_normals.size())
    //     {
    //         face_normals_.resize(face_normals.size());
    //     }

    //     multNx1(M_, face_normals, face_normals_);

    //     // normalize
    // }
}

// void OptixMesh::commit()
// {
//     // build/update acceleration structure
//     if(vertices.size() != vertices_.size())
//     {
//         std::cout << "[OptixMesh::commit()] WARNING: transformation was not applied" << std::endl;
//     }

    

    // const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    // OptixBuildInput triangle_input = {};
    // triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // CUdeviceptr tmp = reinterpret_cast<CUdeviceptr>(vertices_.raw());

    // // VERTICES
    // triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    // triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
    // triangle_input.triangleArray.numVertices   = vertices_.size();
    // triangle_input.triangleArray.vertexBuffers = &tmp;

    // // FACES
    // // std::cout << "- define faces" << std::endl;
    // triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    // triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
    // triangle_input.triangleArray.numIndexTriplets    = faces.size();
    // triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces.raw());

    // // ADDITIONAL SETTINGS
    // triangle_input.triangleArray.flags         = triangle_input_flags;
    // // TODO: this is bad. I define the sbt records inside the sensor programs. 
    // triangle_input.triangleArray.numSbtRecords = 1;

    // // Acceleration Options
    // // Use default options for simplicity.  In a real use case we would want to
    // // enable compaction, etc
    // OptixAccelBuildOptions accel_options = {};

    // unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;

    // { // BUILD FLAGS
    //     build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    //     build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    //     build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    // }

    // accel_options.buildFlags = build_flags;

    // if(m_as)
    // {
    //     // update existing structure
    //     accel_options.operation  = OPTIX_BUILD_OPERATION_UPDATE;
    // } else {
    //     // No acceleration structure exists yet!
    //     m_as = std::make_shared<OptixAccelerationStructure>();
    //     accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    // }

    // OptixAccelBufferSizes gas_buffer_sizes;
    // OPTIX_CHECK( optixAccelComputeMemoryUsage(
    //             m_ctx->ref(),
    //             &accel_options,
    //             &triangle_input,
    //             1, // Number of build inputs
    //             &gas_buffer_sizes
    //             ) );
    
    // CUdeviceptr d_temp_buffer_gas;
    // CUDA_CHECK( cudaMalloc(
    //     reinterpret_cast<void**>( &d_temp_buffer_gas ),
    //     gas_buffer_sizes.tempSizeInBytes) );

    // if(m_as->buffer && (m_as->buffer_size != gas_buffer_sizes.outputSizeInBytes ) )
    // {
    //     CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
    //     m_as->buffer = 0;
    //     m_as->buffer_size = 0;
    // }

    // if(!m_as->buffer)
    // {
    //     CUDA_CHECK( cudaMalloc(
    //             reinterpret_cast<void**>( &m_as->buffer ),
    //             gas_buffer_sizes.outputSizeInBytes
    //             ) );
    //     m_as->buffer_size = gas_buffer_sizes.outputSizeInBytes;
    // }

    // if(!m_stream->context()->isActive())
    // {
    //     m_stream->context()->use();
    // }

    // OPTIX_CHECK( optixAccelBuild(
    //             m_ctx->ref(),
    //             m_stream->handle(),                  // CUDA stream
    //             &accel_options,
    //             &triangle_input,
    //             1,                  // num build inputs
    //             d_temp_buffer_gas,
    //             gas_buffer_sizes.tempSizeInBytes,
    //             m_as->buffer,
    //             gas_buffer_sizes.outputSizeInBytes,
    //             &m_as->handle,
    //             nullptr,            // emitted property list
    //             0                   // num emitted properties
    //             ) );
    
    // // TODO: Compact

    // // // We can now free the scratch space buffer used during build and the vertex
    // // // inputs, since they are not needed by our trivial shading method
    // CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
// }

unsigned int OptixMesh::depth() const
{
    return 0;
}

void OptixMesh::computeFaceNormals()
{
    if(face_normals.size() != faces.size())
    {
        face_normals.resize(faces.size());
    }
    rmagine::computeFaceNormals(vertices, faces, face_normals);
}


const CUdeviceptr* OptixMesh::getVertexBuffer() const
{
    return &m_vertices_ref;
}

CUdeviceptr OptixMesh::getFaceBuffer()
{
    return reinterpret_cast<CUdeviceptr>(faces.raw());
}

OptixMeshPtr make_optix_mesh(const aiMesh* amesh)
{
    OptixMeshPtr ret = std::make_shared<OptixMesh>();

    const aiVector3D* ai_vertices = amesh->mVertices;
    unsigned int num_vertices = amesh->mNumVertices;
    const aiFace* ai_faces = amesh->mFaces;
    unsigned int num_faces = amesh->mNumFaces;

    Memory<Point, RAM> vertices_cpu(num_vertices);
    Memory<Face, RAM> faces_cpu(num_faces);
    Memory<Vector, RAM> face_normals_cpu(num_faces);
    
    // convert
    for(size_t i=0; i<num_vertices; i++)
    {
        vertices_cpu[i] = {
                ai_vertices[i].x,
                ai_vertices[i].y,
                ai_vertices[i].z};
    }
    ret->vertices = vertices_cpu;

    for(size_t i=0; i<num_faces; i++)
    {
        faces_cpu[i].v0 = ai_faces[i].mIndices[0];
        faces_cpu[i].v1 = ai_faces[i].mIndices[1];
        faces_cpu[i].v2 = ai_faces[i].mIndices[2];
    }
    ret->faces = faces_cpu;
    
    for(size_t i=0; i<num_faces; i++)
    {
        unsigned int v0_id = ai_faces[i].mIndices[0];
        unsigned int v1_id = ai_faces[i].mIndices[1];
        unsigned int v2_id = ai_faces[i].mIndices[2];

        const Vector v0{ai_vertices[v0_id].x, ai_vertices[v0_id].y, ai_vertices[v0_id].z};
        const Vector v1{ai_vertices[v1_id].x, ai_vertices[v1_id].y, ai_vertices[v1_id].z};
        const Vector v2{ai_vertices[v2_id].x, ai_vertices[v2_id].y, ai_vertices[v2_id].z};
        
        Vector n = (v1 - v0).normalized().cross((v2 - v0).normalized());
        n.normalize();

        face_normals_cpu[i] = {
            n.x,
            n.y,
            n.z};
    }
    ret->face_normals = face_normals_cpu;

    if(amesh->HasNormals())
    {
        // has vertex normals
        Memory<Vector, RAM> vertex_normals_cpu(num_faces);
        vertex_normals_cpu.resize(num_vertices);
        for(size_t i=0; i<num_vertices; i++)
        {
            vertex_normals_cpu[i] = convert(amesh->mNormals[i]);
        }
        // upload
        ret->vertex_normals = vertex_normals_cpu;
    }

    return ret;
}

} // namespace rmagine