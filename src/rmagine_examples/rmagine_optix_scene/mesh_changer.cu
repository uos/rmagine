#include "mesh_changer.h"

namespace rmagine
{


__global__ void moveVertices_kernel(
    Vector* vertices,
    unsigned int N,
    Vector3 vec)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        vertices[id] += vec;
    }
}

void moveVertices(
    MemoryView<Vector, VRAM_CUDA>& vertices, 
    const Vector3 vec)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (vertices.size() + blockSize - 1) / blockSize;
    moveVertices_kernel<<<gridSize, blockSize>>>(vertices.raw(), vertices.size(), vec);
}

} // namespace rmagine