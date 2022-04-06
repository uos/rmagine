#include "rmagine/math/math_batched.cuh"

namespace rmagine {

template<typename T>
__global__ void copy_kernel(
    const T* from, 
    T* to, 
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = from[id];
    }
}

template<unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if(blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if(blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template<unsigned int blockSize, typename T>
__global__ void chunk_sums_kernel(
    const T* data, 
    unsigned int chunkSize, 
    T* res)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = chunkSize * blockIdx.x + threadIdx.x;
    const unsigned int rows = (chunkSize + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0;
    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < chunkSize)
        {
            sdata[threadIdx.x] += data[globId + blockSize * i];
        }
    }
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}


template<unsigned int blockSize, typename T>
__global__ void chunk_sums_masked_kernel(
    const T* data,
    const bool* mask, 
    unsigned int chunkSize,
    T* res)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = chunkSize * blockIdx.x + threadIdx.x;
    const unsigned int rows = (chunkSize + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0;

    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < chunkSize)
        {
            if(mask[globId + blockSize * i])
            {
                sdata[threadIdx.x] += data[globId + blockSize * i];
            }
        }
    }
    __syncthreads();

    for(unsigned int s=blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}


template<unsigned int blockSize, typename T>
__global__ void chunk_sums_masked_kernel(
    const T* data,
    const unsigned int* mask, 
    unsigned int chunkSize,
    T* res)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = chunkSize * blockIdx.x + threadIdx.x;
    // old: const unsigned int rows = chunkSize / blockSize;
    const unsigned int rows = (chunkSize + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0;

    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < chunkSize)
        {
            if(mask[globId + blockSize * i] > 0)
            {
                sdata[threadIdx.x] += data[globId + blockSize * i];
            }
        }
    }
    __syncthreads();
    
    for(unsigned int s=blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}

template<unsigned int blockSize, typename T>
__global__ void chunk_sums_masked_kernel(
    const T* data,
    const uint8_t* mask, 
    unsigned int chunkSize,
    T* res)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = chunkSize * blockIdx.x + threadIdx.x;
    const unsigned int rows = (chunkSize + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0;

    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < chunkSize)
        {
            if(mask[globId + blockSize * i] > 0)
            {
                sdata[threadIdx.x] += data[globId + blockSize * i];
            }
        }
    }
    __syncthreads();

    for(unsigned int s=blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}

template<typename T>
void chunk_sums(
    const T* data_d, 
    T* res_d, 
    unsigned int Nchunks, 
    unsigned int chunkSize)
{
    if(chunkSize >= 1024) {
        chunk_sums_kernel<1024> <<<Nchunks, 1024>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 512) {
        chunk_sums_kernel<512> <<<Nchunks, 512>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 256) {
        chunk_sums_kernel<256> <<<Nchunks, 256>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 128) {
        chunk_sums_kernel<128> <<<Nchunks, 128>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 64) {
        chunk_sums_kernel<64> <<<Nchunks, 64>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 32) {
        chunk_sums_kernel<32> <<<Nchunks, 32>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 16) {
        chunk_sums_kernel<16> <<<Nchunks, 16>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 8) {
        chunk_sums_kernel<8> <<<Nchunks, 8>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 4) {
        chunk_sums_kernel<4> <<<Nchunks, 4>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 2) {
        chunk_sums_kernel<2> <<<Nchunks, 2>>>(data_d, chunkSize, res_d);
    } else if(chunkSize >= 1) {
        // copy
        constexpr unsigned int blockSize = 64;
        const unsigned int N = Nchunks * chunkSize;
        const unsigned int gridSize = (N + blockSize - 1) / blockSize;
        copy_kernel<<<gridSize, blockSize>>>(data_d, res_d, N);
    }
}

template<typename T>
void chunk_sums_masked(const T* data_d, const bool* mask_d, T* res_d, unsigned int Nchunks, unsigned int chunkSize)
{
    if(chunkSize >= 1024) {
        chunk_sums_masked_kernel<1024> <<<Nchunks, 1024>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 512) {
        chunk_sums_masked_kernel<512> <<<Nchunks, 512>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 256) {
        chunk_sums_masked_kernel<256> <<<Nchunks, 256>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 128) {
        chunk_sums_masked_kernel<128> <<<Nchunks, 128>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 64) {
        chunk_sums_masked_kernel<64> <<<Nchunks, 64>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 32) {
        chunk_sums_masked_kernel<32> <<<Nchunks, 32>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 16) {
        chunk_sums_masked_kernel<16> <<<Nchunks, 16>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 8) {
        chunk_sums_masked_kernel<8> <<<Nchunks, 8>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 4) {
        chunk_sums_masked_kernel<4> <<<Nchunks, 4>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 2) {
        chunk_sums_masked_kernel<2> <<<Nchunks, 2>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 1) {
        std::cout << "WARNING: masked batchedSum with chunkSize 1 called." << std::endl;
        constexpr unsigned int blockSize = 64;
        const unsigned int N = Nchunks * chunkSize;
        const unsigned int gridSize = (N + blockSize - 1) / blockSize;
        copy_kernel<<<gridSize, blockSize>>>(data_d, res_d, N);
    }
}

template<typename T>
void chunk_sums_masked(const T* data_d, const unsigned int* mask_d, T* res_d, unsigned int Nchunks, unsigned int chunkSize)
{
    if(chunkSize >= 1024) {
        chunk_sums_masked_kernel<1024> <<<Nchunks, 1024>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 512) {
        chunk_sums_masked_kernel<512> <<<Nchunks, 512>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 256) {
        chunk_sums_masked_kernel<256> <<<Nchunks, 256>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 128) {
        chunk_sums_masked_kernel<128> <<<Nchunks, 128>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 64) {
        chunk_sums_masked_kernel<64> <<<Nchunks, 64>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 32) {
        chunk_sums_masked_kernel<32> <<<Nchunks, 32>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 16) {
        chunk_sums_masked_kernel<16> <<<Nchunks, 16>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 8) {
        chunk_sums_masked_kernel<8> <<<Nchunks, 8>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 4) {
        chunk_sums_masked_kernel<4> <<<Nchunks, 4>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 2) {
        chunk_sums_masked_kernel<2> <<<Nchunks, 2>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 1) {
        std::cout << "WARNING: masked batchedSum with chunkSize 1 called." << std::endl;
        constexpr unsigned int blockSize = 64;
        const unsigned int N = Nchunks * chunkSize;
        const unsigned int gridSize = (N + blockSize - 1) / blockSize;
        copy_kernel<<<gridSize, blockSize>>>(data_d, res_d, N);
    }
}

template<typename T>
void chunk_sums_masked(const T* data_d, const uint8_t* mask_d, T* res_d, unsigned int Nchunks, unsigned int chunkSize)
{
    if(chunkSize >= 1024) {
        chunk_sums_masked_kernel<1024> <<<Nchunks, 1024>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 512) {
        chunk_sums_masked_kernel<512> <<<Nchunks, 512>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 256) {
        chunk_sums_masked_kernel<256> <<<Nchunks, 256>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 128) {
        chunk_sums_masked_kernel<128> <<<Nchunks, 128>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 64) {
        chunk_sums_masked_kernel<64> <<<Nchunks, 64>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 32) {
        chunk_sums_masked_kernel<32> <<<Nchunks, 32>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 16) {
        chunk_sums_masked_kernel<16> <<<Nchunks, 16>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 8) {
        chunk_sums_masked_kernel<8> <<<Nchunks, 8>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 4) {
        chunk_sums_masked_kernel<4> <<<Nchunks, 4>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 2) {
        chunk_sums_masked_kernel<2> <<<Nchunks, 2>>>(data_d, mask_d, chunkSize, res_d);
    } else if(chunkSize >= 1) {
        std::cout << "WARNING: masked batchedSum with chunkSize 1 called." << std::endl;
        constexpr unsigned int blockSize = 64;
        const unsigned int N = Nchunks * chunkSize;
        const unsigned int gridSize = (N + blockSize - 1) / blockSize;
        copy_kernel<<<gridSize, blockSize>>>(data_d, res_d, N);
    }
}

//////////
// #sumBatched
void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    Memory<Vector, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums<Vector>(data.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    size_t batchSize)
{
    const size_t Nchunks = data.size() / batchSize;
    Memory<Vector, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, sums);
    return sums;
}

void sumBatched(
    const Memory<Matrix3x3, VRAM_CUDA>& data,
    Memory<Matrix3x3, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums<Matrix3x3>(data.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<Matrix3x3, VRAM_CUDA> sumBatched(
    const Memory<Matrix3x3, VRAM_CUDA>& data,
    size_t batchSize)
{
    size_t Nchunks = data.size() / batchSize;
    Memory<Matrix3x3, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, sums);
    return sums;
}

void sumBatched(
    const Memory<float, VRAM_CUDA>& data,
    Memory<float, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums<float>(data.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<float, VRAM_CUDA> sumBatched(
    const Memory<float, VRAM_CUDA>& data,
    size_t batchSize)
{
    size_t Nchunks = data.size() / batchSize;
    Memory<float, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, sums);
    return sums;
}

//////////
// #sumBatched masked
void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<bool, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums_masked<Vector>(data.raw(), mask.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<bool, VRAM_CUDA>& mask,
    size_t batchSize)
{
    size_t Nchunks = data.size() / batchSize;
    Memory<Vector, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, mask, batchSize);
    return sums;
}

void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<unsigned int, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums_masked<Vector>(data.raw(), mask.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<unsigned int, VRAM_CUDA>& mask,
    size_t batchSize)
{
    size_t Nchunks = data.size() / batchSize;
    Memory<Vector, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, mask, batchSize);
    return sums;
}

void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<uint8_t, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums)
{
    const size_t Nchunks = sums.size();
    const size_t batchSize = data.size() / Nchunks;
    chunk_sums_masked<Vector>(data.raw(), mask.raw(), sums.raw(), Nchunks, batchSize);
}

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<uint8_t, VRAM_CUDA>& mask,
    size_t batchSize)
{
    size_t Nchunks = data.size() / batchSize;
    Memory<Vector, VRAM_CUDA> sums(Nchunks);
    sumBatched(data, mask, batchSize);
    return sums;
}

////////
// #covBatched
void covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& covs)
{
    const size_t Nchunks = covs.size();
    const size_t batchSize = m1.size() / Nchunks;
    Memory<Matrix3x3, VRAM_CUDA> cov_parts(m1.size());
    multNxNTransposed(m1, m2, cov_parts);

    sumBatched(cov_parts, covs);
    const unsigned int tmp = batchSize; 
    divNx1Inplace(covs, tmp);
}

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    unsigned int batchSize)
{
    size_t Nchunks = m1.size() / batchSize;
    Memory<Matrix3x3, VRAM_CUDA> covs(Nchunks);
    covBatched(m1, m2, covs);
    return covs;
}

void covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& corr,
    const Memory<unsigned int, VRAM_CUDA>& ncorr,
    Memory<Matrix3x3, VRAM_CUDA>& covs)
{
    const size_t Nchunks = covs.size();
    const size_t batchSize = m1.size() / Nchunks;
    Memory<Matrix3x3, VRAM_CUDA> cov_parts(m1.size());
    multNxNTransposed(m1, m2, corr, cov_parts);
    sumBatched(cov_parts, covs);
    divNxNInplace(covs, ncorr);
}

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& corr,
    const Memory<unsigned int, VRAM_CUDA>& ncorr,
    unsigned int batchSize)
{
    size_t Nchunks = m1.size() / batchSize;
    Memory<Matrix3x3, VRAM_CUDA> covs(Nchunks);
    covBatched(m1, m2, corr, ncorr, covs);
    return covs;
}

} // namespace rmagine