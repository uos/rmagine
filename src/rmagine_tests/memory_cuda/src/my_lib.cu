#include "my_lib.cuh"

__global__ void add_kernel(
    const float* a, 
    const float* b,
    float* res, 
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        res[id] = a[id] + b[id];
    }
}

void add(const rm::MemoryView<float, rm::VRAM_CUDA>& a,
    const rm::MemoryView<float, rm::VRAM_CUDA>& b,
    rm::MemoryView<float, rm::VRAM_CUDA>& res)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (a.size() + blockSize - 1) / blockSize;
    add_kernel<<<gridSize, blockSize>>>(a.raw(), b.raw(), res.raw(), a.size());
}

rm::Memory<float, rm::VRAM_CUDA> add(
    const rm::MemoryView<float, rm::VRAM_CUDA>& a,
    const rm::MemoryView<float, rm::VRAM_CUDA>& b)
{
    rm::Memory<float, rm::VRAM_CUDA> res(a.size());
    add(a, b, res);
    return res;
}