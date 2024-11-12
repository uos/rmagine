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

template<unsigned int nMemElems>
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
    __shared__ CrossStatistics sdata[nMemElems];

    const unsigned int n_threads = blockDim.x;
    const unsigned int n_blocks = gridDim.x;

    const unsigned int total_threads = n_threads * n_blocks;
    const unsigned int n_rows = (N + total_threads - 1) / total_threads;
    const unsigned int n_elems_per_block = n_rows * nMemElems;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int glob_shift = n_elems_per_block * bid;
    
    // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    // reshape:
    // --- blockSize ---
    // [1,  4,  7, 10]  |
    // [2,  5,  8, 11]  | rows
    // [3,  6,  9, 12]  |
    //  |   |   |   |     init reduce to shared mem
    // [6, 15, 24, 33]
    // [21, 57]
    // [78]

    CrossStatistics cross_stats = CrossStatistics::Identity();
    sdata[tid] = CrossStatistics::Identity();

    for(unsigned int i=0; i<n_rows; i++)
    {
        const unsigned int data_id = glob_shift + i * nMemElems + tid; // advance one row
        // TODO: check if this if-statement is correct:
        // before 'tid + blockSize * i < N' but I think that's wrong
        if(data_id < N)
        {
            if(    (dataset_mask == NULL || dataset_mask[data_id] > 0)
                && (model_mask == NULL   || model_mask[data_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[data_id] == params.dataset_id)
                && (model_ids == NULL    || model_ids[data_id]   == params.model_id)
                )
            {
                const Vector Di = pre_transform * dataset_points[data_id]; // read
                const Vector Mi = model_points[data_id]; // read

                const float dist = (Mi - Di).l2normSquared();

                if(dist < params.max_dist * params.max_dist)
                {
                    // cross_stats += CrossStatistics::Init(Di, Mi);
                    sdata[tid] += CrossStatistics::Init(Di, Mi);
                }
            }
        }
    }
    // sdata[tid] = cross_stats;
    __syncthreads();

    for(unsigned int s = nMemElems / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        res[bid] = sdata[0];
    }
}

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats)
{
    const unsigned int n_outputs = stats.size(); // also number of blocks
    constexpr unsigned int n_threads = 1024; // also shared mem

    statistics_p2p_kernel<n_threads> <<<n_outputs, n_threads>>>(
        dataset.points.raw(), dataset.mask.raw(), dataset.ids.raw(), 
        pre_transform,
        model.points.raw(), model.mask.raw(), model.ids.raw(),
        params,
        dataset.points.size(),
        stats.raw()
        );
}

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
    // create a memory view on existing RAM
    MemoryView<CrossStatistics, RAM> stats_view(&stats, 1);
    // to upload it to GPU
    Memory<CrossStatistics, VRAM_CUDA> stats_gpu = stats_view;
    // to write results to it
    statistics_p2p(pre_transform, dataset, model, params, stats_gpu);
    // download to view and therefore update 'stats' with it
    stats_view = stats_gpu;
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


template<unsigned int nMemElems>
__global__ void statistics_p2l_kernel(
    const Vector*   dataset_points,
    const unsigned int* dataset_mask,
    const unsigned int* dataset_ids,
    const Transform pre_transform,
    const Vector*   model_points,
    const Vector*   model_normals,
    const unsigned int* model_mask,
    const unsigned int* model_ids,
    const UmeyamaReductionConstraints params,
    unsigned int N,
    CrossStatistics* res)
{
    __shared__ CrossStatistics sdata[nMemElems];

    const unsigned int n_threads = blockDim.x;
    const unsigned int n_blocks = gridDim.x;

    const unsigned int total_threads = n_threads * n_blocks;
    const unsigned int n_rows = (N + total_threads - 1) / total_threads;
    const unsigned int n_elems_per_block = n_rows * nMemElems;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int glob_shift = n_elems_per_block * bid;
    
    // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    // reshape:
    // --- blockSize ---
    // [1,  4,  7, 10]  |
    // [2,  5,  8, 11]  | rows
    // [3,  6,  9, 12]  |
    //  |   |   |   |     init reduce to shared mem
    // [6, 15, 24, 33]
    // [21, 57]
    // [78]

    CrossStatistics cross_stats = CrossStatistics::Identity();
    sdata[tid] = CrossStatistics::Identity();

    for(unsigned int i=0; i<n_rows; i++)
    {
        const unsigned int data_id = glob_shift + i * nMemElems + tid; // advance one row
        // TODO: check if this if-statement is correct:
        // before 'tid + blockSize * i < N' but I think that's wrong
        if(data_id < N)
        {
            if(    (dataset_mask == NULL || dataset_mask[data_id] > 0)
                && (model_mask == NULL   || model_mask[data_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[data_id] == params.dataset_id)
                && (model_ids == NULL    || model_ids[data_id]   == params.model_id)
                )
            {
                const Vector Di = pre_transform * dataset_points[i]; // read
                const Vector Ii = model_points[i]; // read
                const Vector Ni = model_normals[i];

                const float signed_plane_dist = (Ii - Di).dot(Ni);

                if(fabs(signed_plane_dist) < params.max_dist)
                {
                    // nearest point on model
                    const Vector Mi = Di + Ni * signed_plane_dist;
                    // add Di -> Mi correspondence
                    sdata[tid] += CrossStatistics::Init(Di, Mi);
                }
            }
        }
    }
    // sdata[tid] = cross_stats;
    __syncthreads();

    for(unsigned int s = nMemElems / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        res[bid] = sdata[0];
    }
}


void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats)
{
    const unsigned int n_outputs = stats.size(); // also number of blocks
    constexpr unsigned int n_threads = 1024; // also shared mem

    statistics_p2l_kernel<n_threads> <<<n_outputs, n_threads>>>(
        dataset.points.raw(), dataset.mask.raw(), dataset.ids.raw(), 
        pre_transform,
        model.points.raw(), model.normals.raw(), model.mask.raw(), model.ids.raw(),
        params,
        dataset.points.size(),
        stats.raw()
        );
}

void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
    // create a memory view on existing RAM
    MemoryView<CrossStatistics, RAM> stats_view(&stats, 1);
    // to upload it to GPU
    Memory<CrossStatistics, VRAM_CUDA> stats_gpu = stats_view;
    // to write results to it
    statistics_p2l(pre_transform, dataset, model, params, stats_gpu);
    // download to view and therefore update 'stats' with it
    stats_view = stats_gpu;
}

CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params)
{
    CrossStatistics ret = CrossStatistics::Identity();
    statistics_p2l(pre_transform, dataset, model, params, ret);
    return ret;
}

} // namespace rmagine