#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/cuda/CudaStream.hpp"

#include <cuda_runtime.h>


namespace rmagine {

// CUDA HELPER
namespace cuda {

void* memcpyHostToDevice(void* dest, const void* src, std::size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyHostToDevice) );
    return dest;
}

void* memcpyHostToDevice(void* dest, const void* src, std::size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice, stream->handle()) );
    return dest;
}

void* memcpyDeviceToHost(void* dest, const void* src, std::size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToHost) );
    return dest;
}

void* memcpyDeviceToHost(void* dest, const void* src, std::size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost, stream->handle()) );
    return dest;
}

void* memcpyDeviceToDevice(void* dest, const void* src, std::size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice) );
    return dest;
}

void* memcpyDeviceToDevice( void* dest, const void* src, std::size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, stream->handle()) );
    return dest;
}

void* memcpyHostToHost(void* dest, const void* src, std::size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyHostToHost) );
    return dest;
}

void* memcpyHostToHost(     void* dest, const void* src, std::size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToHost, stream->handle()) );
    return dest;
}

} // namespace cuda

} // namespace mamcl