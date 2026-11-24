/**
 * @file HiprtScene.cpp
 * @brief HIPRT scene implementation
 */

#include "rmagine/map/hiprt/HiprtScene.hpp"

#include <hiprt/hiprt.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace rmagine
{

HiprtScene::HiprtScene(HiprtContextPtr ctx)
    : m_ctx(ctx)
{
}

HiprtScene::~HiprtScene()
{
    if (m_geom) {
        hiprtDestroyGeometry(m_ctx->handle(), m_geom);
        m_geom = nullptr;
    }

    if (m_mesh_data_device) {
        hipFree(m_mesh_data_device);
        m_mesh_data_device = nullptr;
    }
}

unsigned int HiprtScene::add(HiprtMeshPtr mesh)
{
    unsigned int id = m_next_id++;
    mesh->id = id;
    m_meshes[id] = mesh;
    m_needs_rebuild = true;
    return id;
}

unsigned long long HiprtScene::handle() const
{
    return reinterpret_cast<unsigned long long>(m_geom);
}

void HiprtScene::commit()
{
    if (!m_needs_rebuild && m_geom) {
        return;  // No changes
    }

    const size_t n_meshes = m_meshes.size();
    if (n_meshes == 0) {
        std::cerr << "[HiprtScene::commit] Warning: no meshes to build" << std::endl;
        return;
    }

    // Destroy old geometry if exists
    if (m_geom) {
        hiprtDestroyGeometry(m_ctx->handle(), m_geom);
        m_geom = nullptr;
    }

    // For simplicity, we merge all meshes into one triangle geometry
    // (This matches how OptixScene::buildGAS works for a single-mesh case)
    // TODO: Support multiple meshes with proper SBT-like indexing

    // Count total vertices and faces
    size_t total_vertices = 0;
    size_t total_faces = 0;
    for (auto& [id, mesh] : m_meshes) {
        total_vertices += mesh->vertices.size();
        total_faces += mesh->faces.size();
    }

    // Merge all meshes into single buffers
    // (For proof-of-concept; proper multi-mesh needs per-mesh tracking)
    std::vector<float> all_vertices;
    std::vector<uint32_t> all_indices;
    all_vertices.reserve(total_vertices * 3);
    all_indices.reserve(total_faces * 3);

    size_t vertex_offset = 0;
    for (auto& [id, mesh] : m_meshes) {
        mesh->commit();

        // Copy vertices (download to host, merge, upload)
        Memory<Point, RAM> verts_host = mesh->vertices;
        for (size_t i = 0; i < verts_host.size(); ++i) {
            all_vertices.push_back(verts_host[i].x);
            all_vertices.push_back(verts_host[i].y);
            all_vertices.push_back(verts_host[i].z);
        }

        // Copy faces with offset
        Memory<Face, RAM> faces_host = mesh->faces;
        for (size_t i = 0; i < faces_host.size(); ++i) {
            all_indices.push_back(faces_host[i].v0 + vertex_offset);
            all_indices.push_back(faces_host[i].v1 + vertex_offset);
            all_indices.push_back(faces_host[i].v2 + vertex_offset);
        }

        vertex_offset += verts_host.size();
    }

    // Upload merged buffers to device
    float* d_vertices = nullptr;
    uint32_t* d_indices = nullptr;

    hipMalloc(&d_vertices, all_vertices.size() * sizeof(float));
    hipMalloc(&d_indices, all_indices.size() * sizeof(uint32_t));

    hipMemcpy(d_vertices, all_vertices.data(), all_vertices.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_indices, all_indices.data(), all_indices.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Create HIPRT triangle mesh primitive
    hiprtTriangleMeshPrimitive mesh_prim;
    mesh_prim.vertices = d_vertices;
    mesh_prim.vertexCount = total_vertices;
    mesh_prim.vertexStride = sizeof(float) * 3;

    mesh_prim.triangleIndices = d_indices;
    mesh_prim.triangleCount = total_faces;
    mesh_prim.triangleStride = sizeof(uint32_t) * 3;

    // Build geometry input
    hiprtGeometryBuildInput geom_input;
    geom_input.type = hiprtPrimitiveTypeTriangleMesh;
    geom_input.primitive.triangleMesh = mesh_prim;
    geom_input.geomType = 0;  // single geometry type

    // Build options - must zero-initialize (batchBuildMaxPrimCount defaults to 0)
    hiprtBuildOptions build_options = {};
    build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    std::cout << "[HiprtScene] Creating geometry: " << total_vertices << " verts, "
              << total_faces << " tris, stride=" << mesh_prim.vertexStride << std::endl;

    // Create geometry
    hiprtError err = hiprtCreateGeometry(
        m_ctx->handle(),
        geom_input,
        build_options,
        m_geom);

    if (err != hiprtSuccess) {
        hipFree(d_vertices);
        hipFree(d_indices);
        throw std::runtime_error("HiprtScene: hiprtCreateGeometry failed with error " + std::to_string(err));
    }

    std::cout << "[HiprtScene] Geometry created, getting temp buffer size..." << std::endl;

    // Get temporary buffer size for BVH build
    size_t temp_size = 0;
    err = hiprtGetGeometryBuildTemporaryBufferSize(
        m_ctx->handle(),
        geom_input,
        build_options,
        temp_size);

    if (err != hiprtSuccess) {
        hipFree(d_vertices);
        hipFree(d_indices);
        throw std::runtime_error("HiprtScene: hiprtGetGeometryBuildTemporaryBufferSize failed with error " + std::to_string(err));
    }

    std::cout << "[HiprtScene] Temp buffer size: " << temp_size << " bytes" << std::endl;

    // Allocate temp buffer if needed
    hiprtDevicePtr temp_buffer = nullptr;
    if (temp_size > 0) {
        hipMalloc(&temp_buffer, temp_size);
    }

    // Build BVH
    err = hiprtBuildGeometry(
        m_ctx->handle(),
        hiprtBuildOperationBuild,
        geom_input,
        build_options,
        temp_buffer,
        nullptr,  // stream (use default)
        m_geom);

    // Free temp buffer
    if (temp_buffer) {
        hipFree(temp_buffer);
    }

    if (err != hiprtSuccess) {
        hipFree(d_vertices);
        hipFree(d_indices);
        throw std::runtime_error("HiprtScene: hiprtBuildGeometry failed with error " + std::to_string(err));
    }

    // Note: We're keeping d_vertices and d_indices alive by not freeing them
    // In a proper implementation, these would be managed by the scene
    // For proof-of-concept, we leak them (they're needed during tracing)

    // Build scene data for trace kernel
    m_scene_data.n_geometries = n_meshes;

    // Upload mesh data
    if (m_mesh_data_device) {
        hipFree(m_mesh_data_device);
    }
    hipMalloc(&m_mesh_data_device, n_meshes * sizeof(HiprtMeshData));

    std::vector<HiprtMeshData> mesh_data_host(n_meshes);
    size_t idx = 0;
    for (auto& [id, mesh] : m_meshes) {
        mesh_data_host[idx] = mesh->mesh_data;
        ++idx;
    }
    hipMemcpy(m_mesh_data_device, mesh_data_host.data(), n_meshes * sizeof(HiprtMeshData), hipMemcpyHostToDevice);
    m_scene_data.geometries = m_mesh_data_device;

    m_needs_rebuild = false;

    std::cout << "[HiprtScene::commit] Built BVH with " << total_vertices << " vertices, "
              << total_faces << " faces" << std::endl;
}

} // namespace rmagine
