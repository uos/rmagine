#include "MemoryCuda.hpp"

namespace rmagine 
{

/// VRAM_CUDA
template<typename DataT>
DataT* VRAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    CUDA_DEBUG( cudaMalloc(&ret, N * sizeof(DataT)) );
    return ret;
}

template<typename DataT>
DataT* VRAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    CUDA_DEBUG( cudaMalloc(&ret, sizeof(DataT) * Nnew) );
    CUDA_DEBUG( cudaFree(mem) );
    return ret;
}

template<typename DataT>
void VRAM_CUDA::free(DataT* mem, size_t N)
{
    // std::cout << "FREE VRAM_CUDA" << std::endl;
    // std::cout << "Size: " << N << std::endl;
    // std::cout << "Type: " << boost::typeindex::type_id<DataT>().pretty_name() << std::endl;
    CUDA_DEBUG( cudaFree(mem) );
}


/// RAM_CUDA
template<typename DataT>
DataT* RAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    CUDA_DEBUG( cudaMallocHost(&ret, N * sizeof(DataT) ) );
    return ret;
}

template<typename DataT>
DataT* RAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    CUDA_DEBUG( cudaMallocHost(&ret, Nnew * sizeof(DataT) ) );
    CUDA_DEBUG( cudaFreeHost(mem) );
    return ret;
}

template<typename DataT>
void RAM_CUDA::free(DataT* mem, size_t N)
{
    CUDA_DEBUG( cudaFreeHost(mem) );
}

} // namespace rmagine