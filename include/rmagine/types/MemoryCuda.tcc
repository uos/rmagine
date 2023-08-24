#include "MemoryCuda.hpp"

namespace rmagine 
{

/// VRAM_CUDA
template<typename DataT>
DataT* VRAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    cuda::malloc((void**)&ret, N * sizeof(DataT));
    return ret;
}

template<typename DataT>
DataT* VRAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    cuda::malloc((void**)&ret, sizeof(DataT) * Nnew);
    cuda::free(mem);
    return ret;
}

template<typename DataT>
void VRAM_CUDA::free(DataT* mem, size_t N)
{
    cuda::free(mem);
}

/// RAM_CUDA
template<typename DataT>
DataT* RAM_CUDA::alloc(size_t N)
{
    DataT* ret;
    cuda::mallocHost((void**)&ret, N * sizeof(DataT));
    return ret;
}

template<typename DataT>
DataT* RAM_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    cuda::mallocHost((void**)&ret, Nnew * sizeof(DataT));
    cuda::freeHost(mem);
    return ret;
}

template<typename DataT>
void RAM_CUDA::free(DataT* mem, size_t N)
{
    cuda::freeHost(mem);
}

/// VRAM_CUDA
template<typename DataT>
DataT* UNIFIED_CUDA::alloc(size_t N)
{
    DataT* ret;
    cuda::mallocManaged((void**)&ret, sizeof(DataT) * N);
    return ret;
}

template<typename DataT>
DataT* UNIFIED_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret;
    cuda::mallocManaged((void**)&ret, sizeof(DataT) * Nnew);
    cuda::free(mem);
    return ret;
}

template<typename DataT>
void UNIFIED_CUDA::free(DataT* mem, size_t N)
{
    cuda::free(mem);
}


} // namespace rmagine