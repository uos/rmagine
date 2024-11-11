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
__global__ void statistics_p2p_kernel_old(
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

    CrossStatistics cross_stats = CrossStatistics::Identity();
    for(unsigned int i=0; i<rows; i++)
    {
        // TODO: check if this if-statement is correct:
        // before 'tid + blockSize * i < N' but I think that's wrong
        if(tid + blockSize * i < N)
        {
            const unsigned int inner_id = globId + blockSize * i;

            if(    (dataset_mask == NULL || dataset_mask[inner_id] > 0)
                && (model_mask == NULL   || model_mask[inner_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[inner_id] == params.dataset_id)
                && (model_ids == NULL    || model_ids[inner_id]   == params.model_id)
                )
            {
                const Vector Di = pre_transform * dataset_points[inner_id]; // read
                const Vector Mi = model_points[inner_id]; // read

                const float dist = (Mi - Di).l2normSquared();

                if(dist < params.max_dist * params.max_dist)
                {
                    cross_stats += CrossStatistics::Init(Di, Mi);
                }
            }
        }
    }
    sdata[tid] = cross_stats;
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
    __shared__ Matrix3x3    sdata_cov[blockSize];
    __shared__ Vector3      sdata_dataset_mean[blockSize];
    __shared__ Vector3      sdata_model_mean[blockSize];
    __shared__ unsigned int sdata_nmeas[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    Vector d_mean = {0.0, 0.0, 0.0};
    Vector m_mean = {0.0, 0.0, 0.0};
    unsigned int n_corr = 0;
    Matrix3x3 C;
    C.setZeros();

    // CrossStatistics cross_stats = CrossStatistics::Identity();
    for(unsigned int i=0; i<rows; i++)
    {
        // TODO: check if this if-statement is correct:
        // before 'tid + blockSize * i < N' but I think that's wrong
        if(tid + blockSize * i < N)
        {
            const unsigned int inner_id = globId + blockSize * i;

            if(    (dataset_mask == NULL || dataset_mask[inner_id] > 0)
                && (model_mask == NULL   || model_mask[inner_id]   > 0)
                && (dataset_ids == NULL  || dataset_ids[inner_id] == params.dataset_id)
                && (model_ids == NULL    || model_ids[inner_id]   == params.model_id)
                )
            {
                const Vector Di = pre_transform * dataset_points[inner_id]; // read
                const Vector Mi = model_points[inner_id]; // read

                const float N_1 = static_cast<float>(n_corr);
                const float N = static_cast<float>(n_corr + 1);
                const float w1 = N_1 / N;
                const float w2 = 1.0 / N;

                const Vector d_mean_old = d_mean;
                const Vector m_mean_old = m_mean;

                const Vector d_mean_new = d_mean_old * w1 + Di * w2; 
                const Vector m_mean_new = m_mean_old * w1 + Mi * w2;

                const Matrix3x3 P1 = (m_mean_old - m_mean_new).multT(d_mean_old - d_mean_new);
                const Matrix3x3 P2 = (Mi - m_mean_new).multT(Di - d_mean_new);

                const float dist = (Mi - Di).l2normSquared();

                if(dist < params.max_dist * params.max_dist)
                {
                    // write
                    d_mean = d_mean_new;
                    m_mean = m_mean_new;
                    C = (C + P1) * w1 + P2 * w2;
                    n_corr = n_corr + 1;
                }
            }
        }
    }

    sdata_cov[tid] = C;
    sdata_dataset_mean[tid] = d_mean;
    sdata_model_mean[tid] = m_mean;
    sdata_nmeas[tid] = n_corr;

    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            // sdata_cov += sdata_cov[tid + s];
            // sdata[tid] += sdata[tid + s];

            // read
            const Vector cs1_dataset_mean = sdata_dataset_mean[tid];
            const Vector cs1_model_mean = sdata_model_mean[tid];
            const Matrix3x3 cs1_cov = sdata_cov[tid];
            const unsigned int cs1_nmeas = sdata_nmeas[tid];

            const Vector cs2_dataset_mean = sdata_dataset_mean[tid + s];
            const Vector cs2_model_mean = sdata_model_mean[tid + s];
            const Matrix3x3 cs2_cov = sdata_cov[tid + s];
            const unsigned int cs2_nmeas = sdata_nmeas[tid + s];

            // compute
            const unsigned int cs3_nmeas = static_cast<float>(cs1_nmeas + cs2_nmeas);

            const float w1 = static_cast<float>(cs1_nmeas) / static_cast<float>(cs3_nmeas);
            const float w2 = static_cast<float>(cs2_nmeas) / static_cast<float>(cs3_nmeas);

            const Vector cs3_dataset_mean = cs1_dataset_mean * w1 + cs2_dataset_mean * w2; 
            const Vector cs3_model_mean = cs1_model_mean * w1 + cs2_model_mean * w2;

            auto P1 = (cs1_model_mean - cs3_model_mean).multT(cs1_dataset_mean - cs3_dataset_mean);
            auto P2 = (cs2_model_mean - cs3_model_mean).multT(cs2_dataset_mean - cs3_dataset_mean);
            
            // write
            sdata_dataset_mean[tid] = cs3_dataset_mean;
            sdata_model_mean[tid] = cs3_model_mean;
            sdata_cov[tid] = (cs1_cov + P1) * w1 + (cs2_cov + P2) * w2;
            sdata_nmeas[tid] = cs3_nmeas;
        }
        __syncthreads();
    }

    // if(tid < blockSize / 2 && tid < 32)
    // {
    //     warpReduce<blockSize>(sdata, tid);
    // }

    if(tid == 0)
    {
        res[blockIdx.x].dataset_mean = sdata_dataset_mean[0];
        res[blockIdx.x].model_mean = sdata_model_mean[0];
        res[blockIdx.x].covariance = sdata_cov[0];
        res[blockIdx.x].n_meas = sdata_nmeas[0];
    }
}

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats)
{
    statistics_p2p_kernel<512> <<<1, 512>>>(
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
    CrossStatistics& statistics)
{
    // std::cout << "UPLOAD!" << std::endl;

    // upload statistics
    MemoryView<CrossStatistics, RAM> stats_view(&statistics, 1);
    Memory<CrossStatistics, VRAM_CUDA> stats_gpu = stats_view;

    statistics_p2p(pre_transform, dataset, model, params, stats_gpu);

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