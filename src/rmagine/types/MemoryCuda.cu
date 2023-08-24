#include "rmagine/types/MemoryCuda.hpp"
#include "rmagine/util/cuda/CudaStream.hpp"

#include <cuda_runtime.h>
#include "rmagine/util/cuda/CudaDebug.hpp"


namespace rmagine {

// CUDA HELPER
namespace cuda {

void* memcpyHostToDevice(void* dest, const void* src, size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyHostToDevice) );
    return dest;
}

void* memcpyHostToDevice(void* dest, const void* src, size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice, stream->handle()) );
    return dest;
}

void* memcpyDeviceToHost(void* dest, const void* src, size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToHost) );
    return dest;
}

void* memcpyDeviceToHost(void* dest, const void* src, size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost, stream->handle()) );
    return dest;
}

void* memcpyDeviceToDevice(void* dest, const void* src, size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice) );
    return dest;
}

void* memcpyDeviceToDevice( void* dest, const void* src, size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, stream->handle()) );
    return dest;
}

void* memcpyHostToHost(void* dest, const void* src, size_t count)
{
    RM_CUDA_CHECK( cudaMemcpy(dest, src, count, cudaMemcpyHostToHost) );
    return dest;
}

void* memcpyHostToHost(     void* dest, const void* src, size_t count, CudaStreamPtr stream)
{
    RM_CUDA_CHECK( cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToHost, stream->handle()) );
    return dest;
}

void** malloc(void** ptr, size_t count)
{
    RM_CUDA_CHECK( cudaMalloc(ptr, count) );
    return ptr;
}

void** mallocHost(void** ptr, size_t count)
{
    RM_CUDA_CHECK( cudaMallocHost(ptr, count) );
    return ptr;
}

void** mallocManaged(void** ptr, size_t count)
{
    RM_CUDA_CHECK( cudaMallocManaged(ptr, count) );
    return ptr;
}

void* free(void* ptr)
{
    RM_CUDA_CHECK( cudaFree(ptr) );
    return ptr;
}

void* freeHost(void* ptr)
{
    RM_CUDA_CHECK( cudaFreeHost(ptr) );
    return ptr;
}


} // namespace cuda

} // namespace mamcl