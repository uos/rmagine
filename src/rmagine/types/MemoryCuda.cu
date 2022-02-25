#include "rmagine/types/MemoryCuda.hpp"

#include <cuda_runtime.h>

namespace rmagine {

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

} // namespace mamcl