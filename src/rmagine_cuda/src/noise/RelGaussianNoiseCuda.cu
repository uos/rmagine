#include "rmagine/noise/RelGaussianNoiseCuda.hpp"

#include <rmagine/util/cuda/random.cuh>

namespace rmagine
{


__global__ void kernel_rel_gaussian_noise_apply(
    float* data,
    curandState* states,
    unsigned int N, 
    float mean, 
    float stddev,
    float range_exp,
    float max_range)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        curandState state = states[idx];

        const float range = data[idx];

        if(range <= max_range)
        {
            const float stddev_range = stddev * powf(range, range_exp);
            // mean also?
            data[idx] += curand_normal(&state) * stddev_range + mean;
            // update state
            states[idx] = state;
        }
    }
}

RelGaussianNoiseCuda::RelGaussianNoiseCuda(
    float mean, 
    float stddev, 
    float range_exp,
    NoiseCuda::Options options)
:NoiseCuda(options)
,m_mean(mean)
,m_stddev(stddev)
,m_range_exp(range_exp)
{

}

void RelGaussianNoiseCuda::apply(MemoryView<float, VRAM_CUDA>& ranges)
{
    updateStates(ranges);

    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (ranges.size() + blockSize - 1) / blockSize;

    kernel_rel_gaussian_noise_apply<<<gridSize, blockSize>>>(ranges.raw(), m_states.raw(), ranges.size(), m_mean, m_stddev, m_range_exp, m_options.max_range);
}

} // namespace rmagine