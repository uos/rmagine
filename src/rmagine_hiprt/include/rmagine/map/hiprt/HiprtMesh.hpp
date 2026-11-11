/**
 * @file HiprtMesh.hpp
 * @brief Triangle mesh for HIPRT ray tracing
 *
 * Mirrors OptixMesh but uses HIPRT geometry primitives.
 */

#ifndef RMAGINE_MAP_HIPRT_MESH_HPP
#define RMAGINE_MAP_HIPRT_MESH_HPP

#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/mesh_types.h>
#include <rmagine/math/types.h>
#include <rmagine/simulation/hiprt/sim_program_data.h>

namespace rmagine
{

class HiprtMesh;
using HiprtMeshPtr = std::shared_ptr<HiprtMesh>;

/**
 * @brief Triangle mesh geometry for HIPRT
 *
 * Holds vertex/face buffers in device memory and the
 * per-mesh data (normals, ID) for the trace kernel.
 */
class HiprtMesh
{
public:
    HiprtMesh();
    ~HiprtMesh();

    // Non-copyable
    HiprtMesh(const HiprtMesh&) = delete;
    HiprtMesh& operator=(const HiprtMesh&) = delete;

    /**
     * @brief Compute face normals from vertices and faces
     */
    void computeFaceNormals();

    /**
     * @brief Commit mesh data (prepare for BVH build)
     */
    void commit();

    // Vertex buffer (Point = float3)
    Memory<Point, VRAM_CUDA> vertices;

    // Face buffer (Face = uint3)
    Memory<Face, VRAM_CUDA> faces;

    // Per-vertex normals (optional)
    Memory<Vector, VRAM_CUDA> vertex_normals;

    // Per-face normals
    Memory<Vector, VRAM_CUDA> face_normals;

    // Mesh ID (set by scene when added)
    unsigned int id = 0;

    // SBT-like data for trace kernel
    HiprtMeshData mesh_data;

    // Pre-transform (3x4 row-major matrix, device memory)
    float* pre_transform = nullptr;
    float pre_transform_h[12];

    /**
     * @brief Apply the current transform to pre_transform buffer
     */
    void apply(const Matrix4x4& M);
};

} // namespace rmagine

#endif // RMAGINE_MAP_HIPRT_MESH_HPP
