#include "rmagine/map/mesh_preprocessing.cuh"

namespace rmagine
{

__global__ void computeFaceNormals_kernel(
    const Vector3* vertices,
    const Face* faces,
    Vector3* face_normals,
    unsigned int Nfaces)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < Nfaces)
    {
        const Vector v0 = vertices[faces[id].v0];
        const Vector v1 = vertices[faces[id].v1];
        const Vector v2 = vertices[faces[id].v2];
        face_normals[id] = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();
    }
}

void computeFaceNormals(
    const MemoryView<Vector3, VRAM_CUDA>& vertices,
    const MemoryView<Face, VRAM_CUDA>& faces,
    MemoryView<Vector3, VRAM_CUDA>& face_normals)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (faces.size() + blockSize - 1) / blockSize;
    computeFaceNormals_kernel<<<gridSize, blockSize>>>(
        vertices.raw(), 
        faces.raw(), 
        face_normals.raw(), 
        faces.size());
}

Memory<Vector3, VRAM_CUDA> computeFaceNormals(
    const MemoryView<Vector3, VRAM_CUDA>& vertices,
    const MemoryView<Face, VRAM_CUDA>& faces)
{
    Memory<Vector3, VRAM_CUDA> face_normals(faces.size());
    computeFaceNormals(vertices, faces, face_normals);
    return face_normals;
}

} // namespace rmagine