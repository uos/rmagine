#include "imagine/types/MemoryCuda.hpp"

#include <cuda_runtime.h>

namespace imagine {

// CUDA HELPER
namespace cuda {

void* memcpyHostToDevice(void* dest, const void* src, std::size_t count)
{
    CUDA_DEBUG( cudaMemcpy(dest, src, count, cudaMemcpyHostToDevice) );
    return dest;
}

void* memcpyDeviceToHost(void* dest, const void* src, std::size_t count)
{
    CUDA_DEBUG( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToHost) );
    return dest;
}

void* memcpyDeviceToDevice(void* dest, const void* src, std::size_t count)
{
    CUDA_DEBUG( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice) );
    return dest;
}

void* memcpyHostToHost(void* dest, const void* src, std::size_t count)
{
    CUDA_DEBUG( cudaMemcpy(dest, src, count, cudaMemcpyHostToHost) );
    return dest;
}

} // namespace cuda

void* VRAM_CUDA::alloc(size_t N)
{
    void* ret;
    CUDA_DEBUG( cudaMalloc(&ret, N) );
    return ret;
}

void* VRAM_CUDA::realloc(void* mem, size_t N)
{
    void* ret;
    CUDA_DEBUG( cudaMalloc(&ret, N) );
    // what if N smaller then old memory?
    // cudaMemcpy(&ret, mem, N, cudaMemcpyDeviceToDevice);
    CUDA_DEBUG( cudaFree(mem) );
    return ret;
}

void VRAM_CUDA::free(void* mem)
{
    CUDA_DEBUG( cudaFree(mem) );
}

// RAM CUDA
void* RAM_CUDA::alloc(size_t N)
{
    void* ret;
    CUDA_DEBUG( cudaMallocHost(&ret, N) );
    return ret;
}

void* RAM_CUDA::realloc(void* mem, size_t N)
{
    void* ret;
    CUDA_DEBUG( cudaMallocHost(&ret, N) );
    CUDA_DEBUG( cudaFreeHost(mem) );
    return ret;
}

void RAM_CUDA::free(void* mem)
{
    CUDA_DEBUG( cudaFreeHost(mem) );
}

} // namespace mamcl