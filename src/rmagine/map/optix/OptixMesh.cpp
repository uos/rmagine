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

OptixMesh::~OptixMesh()
{
    // std::cout << "[OptixMesh::~OptixMesh()] destroyed." << std::endl;
    if(pre_transform)
    {
        cudaFree( reinterpret_cast<void*>( pre_transform ) );
    }
}

void OptixMesh::apply()
{
    Matrix4x4 M = matrix();

    pre_transform_h[ 0] = M(0,0); // Rxx
    pre_transform_h[ 1] = M(0,1); // Rxy
    pre_transform_h[ 2] = M(0,2); // Rxz
    pre_transform_h[ 3] = M(0,3); // tx
    pre_transform_h[ 4] = M(1,0); // Ryx
    pre_transform_h[ 5] = M(1,1); // Ryy
    pre_transform_h[ 6] = M(1,2); // Ryz
    pre_transform_h[ 7] = M(1,3); // ty 
    pre_transform_h[ 8] = M(2,0); // Rzx
    pre_transform_h[ 9] = M(2,1); // Rzy
    pre_transform_h[10] = M(2,2); // Rzz
    pre_transform_h[11] = M(2,3); // tz

    if(!pre_transform)
    {
        RM_CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>(&pre_transform), sizeof(float) * 12 ) );
    }

    RM_CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>(pre_transform),
        pre_transform_h,
        sizeof(float) * 12,
        cudaMemcpyHostToDevice
    ));

    m_changed = true;
}

void OptixMesh::commit()
{
    m_vertices_ref = reinterpret_cast<CUdeviceptr>(vertices.raw());

    sbt_data.vertex_normals = vertex_normals.raw();
    sbt_data.face_normals = face_normals.raw();
}

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

OptixMeshPtr make_optix_mesh(
    const aiMesh* amesh,
    OptixContextPtr context)
{
    OptixMeshPtr ret = std::make_shared<OptixMesh>(context);

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

    ret->computeFaceNormals();

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

    ret->apply();

    return ret;
}

} // namespace rmagine