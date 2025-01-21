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
    const uint8_t* dataset_mask,
    const unsigned int* dataset_ids,
    const Transform pre_transform,
    const Vector*   model_points,
    const uint8_t* model_mask,
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
    constexpr unsigned int n_threads = 512; // also shared mem

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
    const uint8_t* dataset_mask,
    const unsigned int* dataset_ids,
    const Transform pre_transform,
    const Vector*   model_points,
    const Vector*   model_normals,
    const uint8_t* model_mask,
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
                const Vector Ii = model_points[data_id]; // read
                const Vector Ni = model_normals[data_id];

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

template<unsigned int nMemElems>
__global__ void statistics_objectwise_p2l_kernel(
    const Vector*   dataset_points,
    const uint8_t* dataset_mask,
    const unsigned int* dataset_ids,
    const uint32_t width,
    const uint32_t height,
    const Transform* pre_transforms,
    const Vector*   model_points,
    const Vector*   model_normals,
    const uint8_t* model_mask,
    const unsigned int* model_ids,
    const UmeyamaReductionConstraints* params,
    const AABB* bboxes,
    unsigned int N,
    CrossStatistics* res)
{

    // nMemElems == num_threads!!!!
    __shared__ CrossStatistics sdata[nMemElems];

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int n_threads = blockDim.x;

    const AABB bb = bboxes[bid];
    const unsigned int min_col = bb.min[0];
    const unsigned int min_row = bb.min[1];
    const unsigned int max_col = bb.max[0];
    const unsigned int max_row = bb.max[1];
    const unsigned int bb_width  = max_col - min_col;
    const unsigned int bb_height = max_row - min_row;

    const unsigned int start_idx = min_row * width + min_col; 
    const unsigned int n_elems = bb_width * bb_height;
    const unsigned int n_elems_thread = (n_elems + n_threads - 1) / n_threads;

    const Transform pre_transform = pre_transforms[bid];

    CrossStatistics cross_stats = CrossStatistics::Identity();
    sdata[tid] = CrossStatistics::Identity();
    const UmeyamaReductionConstraints param = params[bid];

    unsigned int t_idx = tid * n_elems_thread; 
    for(unsigned int i=0; i<n_elems_thread; i++)
    {
        unsigned int idx = t_idx + i;
        unsigned int row = idx / bb_width;
        unsigned int col = idx % bb_width;
        // width is the stride from row to row.
        const unsigned int data_id = start_idx + row * width + col; 
 
        if (data_id < N)
        {
            if(    (dataset_mask == NULL || dataset_mask[data_id] > 0)
                && (model_mask == NULL   || model_mask[data_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[data_id] == param.dataset_id)
                && (model_ids == NULL    || model_ids[data_id]   == param.model_id)
                )
            {  
                const Vector Di = pre_transform * dataset_points[data_id]; // read
                const Vector Ii = model_points[data_id]; // read
                const Vector Ni = model_normals[data_id];

                const float signed_plane_dist = (Ii - Di).dot(Ni);

                if(fabs(signed_plane_dist) < param.max_dist)
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
        // printf("num_valid %i", sdata[0].n_meas);
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
    constexpr unsigned int n_threads = 512; // also shared mem

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

void statistics_objectwise_p2l(
    const MemoryView<Transform, VRAM_CUDA>& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const unsigned int& width,
    const unsigned int& height,
    const MemoryView<UmeyamaReductionConstraints, VRAM_CUDA>& params,
    const MemoryView<AABB, VRAM_CUDA>& bboxes,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats)
{
    const unsigned int n_outputs = stats.size(); // also number of blocks
    constexpr unsigned int n_threads = 512; // also shared mem

    statistics_objectwise_p2l_kernel<n_threads> <<<n_outputs, n_threads>>>(
        dataset.points.raw(), dataset.mask.raw(), dataset.ids.raw(), 
        width, height,
        pre_transform.raw(),
        model.points.raw(), model.normals.raw(), model.mask.raw(), model.ids.raw(),
        params.raw(),
        bboxes.raw(),
        dataset.points.size(),
        stats.raw()
        );
}

void statistics_objectwise_p2l(
    const MemoryView<Transform, RAM>& pre_transforms,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const unsigned int& width,
    const unsigned int& height,
    const MemoryView<UmeyamaReductionConstraints, RAM>& params,
    const MemoryView<AABB, RAM>& bboxes,
    MemoryView<CrossStatistics, RAM>& stats)
{
    // Upload it to GPU
    Memory<CrossStatistics, VRAM_CUDA> stats_gpu = stats;
    Memory<Transform, VRAM_CUDA> pre_transforms_gpu = pre_transforms;
    Memory<UmeyamaReductionConstraints, VRAM_CUDA> params_gpu = params;
    Memory<AABB, VRAM_CUDA> bboxes_gpu = bboxes;

    // to write results to it
    statistics_objectwise_p2l(pre_transforms_gpu, dataset, model, width, height,
     params_gpu, bboxes_gpu, stats_gpu);
    // download to view and therefore update 'stats' with it
    stats = stats_gpu;
}

} // namespace rmagine