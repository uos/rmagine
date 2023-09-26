#include "rmagine/util/cuda/random.cuh"
#include <iostream>

namespace rmagine
{

__global__ void kernel_curand_init(
    curandState* states, 
    unsigned int N,
    unsigned int seed = 42)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

void random_init(
    MemoryView<curandState, VRAM_CUDA>& states,
    unsigned int seed)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (states.size() + blockSize - 1) / blockSize;

    kernel_curand_init<<<gridSize, blockSize>>>(states.raw(), states.size(), seed);
}

} // namespace rmagine