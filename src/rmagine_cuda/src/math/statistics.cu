#include "rmagine/math/statistics.cuh"

namespace rmagine {

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
__global__ void sum_kernel(
    const T* data,
    T* res,
    unsigned int N)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0; // TODO: this is a trick, but not good
    for(unsigned int i=0; i<rows; i++)
    {
        if(globId + blockSize * i < N)
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

template<unsigned int blockSize>
__global__ void statistics_p2p_kernel(
    const Vector*   dataset_points,
    const unsigned int* dataset_mask,
    const unsigned int* dataset_ids,
    const Transform pre_transform,
    const Vector*   model_points,
    const unsigned int* model_mask,
    const unsigned int* model_ids,
    const UmeyamaReductionConstraints params,
    unsigned int N,
    CrossStatistics* res)
{
    __shared__ CrossStatistics sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    sdata[tid] = CrossStatistics::Identity();
    for(unsigned int i=0; i<rows; i++)
    {
        // TODO: check if this if-statement is correct:
        // before 'tid + blockSize * i < N' but I think that's wrong
        if(globId + blockSize * i < N)
        {
            const unsigned int inner_id = globId + blockSize * i;

            if(    (dataset_mask == NULL || dataset_mask[inner_id] > 0)
                && (model_mask == NULL   || model_mask[inner_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[inner_id] == params.dataset_id)
                && (model_ids == NULL    || model_ids[inner_id]   == params.model_id)
                )
            {
                const Vector Di = pre_transform * dataset_points[i]; // read
                const Vector Mi = model_points[i]; // read

                const float dist = (Mi - Di).l2norm();

                if(fabs(dist) < params.max_dist)
                {
                    sdata[threadIdx.x] += CrossStatistics::Init(dataset_points[inner_id], model_points[inner_id]);
                }
            }
        }
    }
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // if(tid < blockSize / 2 && tid < 32)
    // {
    //     warpReduce<blockSize>(sdata, tid);
    // }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}


void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& statistics)
{
    // std::cout << "UPLOAD!" << std::endl;

    // upload statistics
    MemoryView<CrossStatistics, RAM> stats_view(&statistics, 1);
    Memory<CrossStatistics, VRAM_CUDA> stats = stats_view;

    statistics_p2p_kernel<512> <<<1, 512>>>(
        dataset.points.raw(), dataset.mask.raw(), dataset.ids.raw(), 
        pre_transform,
        model.points.raw(), model.mask.raw(), model.ids.raw(),
        params,
        dataset.points.size(),
        stats.raw()
        );

    stats_view = stats;
}

CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params)
{
    CrossStatistics ret = CrossStatistics::Identity();
    
    statistics_p2p(pre_transform, dataset, model, params, ret);

    return ret;
}

void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& statistics)
{
    
}

CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params)
{
    CrossStatistics ret = CrossStatistics::Identity();
    
    return ret;
}

} // namespace rmagine