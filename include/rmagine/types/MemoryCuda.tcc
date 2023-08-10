#include "MemoryCuda.hpp"

namespace rmagine 
{

/// VRAM_CUDA
template<typename DataT>
DataT* VRAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMalloc(&ret, N * sizeof(DataT)) );
    return ret;
}

template<typename DataT>
DataT* VRAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMalloc(&ret, sizeof(DataT) * Nnew) );
    RM_CUDA_CHECK( cudaFree(mem) );
    return ret;
}

template<typename DataT>
void VRAM_CUDA::free(DataT* mem, size_t N)
{
    RM_CUDA_CHECK( cudaFree(mem) );
}

/// RAM_CUDA
template<typename DataT>
DataT* RAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMallocHost(&ret, N * sizeof(DataT) ) );
    return ret;
}

template<typename DataT>
DataT* RAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMallocHost(&ret, Nnew * sizeof(DataT) ) );
    RM_CUDA_CHECK( cudaFreeHost(mem) );
    return ret;
}

template<typename DataT>
void RAM_CUDA::free(DataT* mem, size_t N)
{
    RM_CUDA_CHECK( cudaFreeHost(mem) );
}



/// VRAM_CUDA
template<typename DataT>
DataT* UNIFIED_CUDA::alloc(size_t N)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMallocManaged(&ret, N * sizeof(DataT)) );
    return ret;
}

template<typename DataT>
DataT* UNIFIED_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    RM_CUDA_CHECK( cudaMallocManaged(&ret, sizeof(DataT) * Nnew) );
    RM_CUDA_CHECK( cudaFree(mem) );
    return ret;
}

template<typename DataT>
void UNIFIED_CUDA::free(DataT* mem, size_t N)
{
    RM_CUDA_CHECK( cudaFree(mem) );
}


} // namespace rmagine