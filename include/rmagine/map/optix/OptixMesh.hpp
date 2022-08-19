#ifndef RMAGINE_MAP_OPTIX_MESH_HPP
#define RMAGINE_MAP_OPTIX_MESH_HPP

#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <rmagine/util/cuda/CudaContext.hpp>
#include <rmagine/util/optix/OptixContext.hpp>

#include <memory>

#include <assimp/mesh.h>

#include "OptixGeometry.hpp"

#include "optix_definitions.h"

namespace rmagine
{

/**
//  * @brief Single mesh. 
//  * - Cuda Buffers for vertices, faces, vertex_normals and face_normals
//  * - TraversableHandle for raytracing
//  */
class OptixMesh 
: public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixMesh(OptixContextPtr context = optix_default_context());
    OptixMesh(const aiMesh* amesh, OptixContextPtr context = optix_default_context());


    virtual ~OptixMesh();

    virtual void apply();
    // virtual void commit();

    virtual unsigned int depth() const;

    virtual OptixGeometryType type() const 
    {
        return OptixGeometryType::MESH;
    }

    void computeFaceNormals();

    const CUdeviceptr* getVertexBuffer() const;
    CUdeviceptr getFaceBuffer();

    // TODO manage read and write access over functions
    // before transform: write here
    Memory<Point, VRAM_CUDA>    vertices;
    Memory<Face, VRAM_CUDA>     faces;
    Memory<Vector, VRAM_CUDA>   face_normals;
    Memory<Vector, VRAM_CUDA>   vertex_normals;
    

    // after transform: read here
    Memory<Point, VRAM_CUDA>    vertices_;
    Memory<Vector, VRAM_CUDA>   face_normals_;
    Memory<Vector, VRAM_CUDA>   vertex_normals_;

private:
    CUdeviceptr m_vertices_ref;
};

using OptixMeshPtr = std::shared_ptr<OptixMesh>;

OptixMeshPtr make_optix_mesh(const aiMesh* amesh);

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_MESH_HPP