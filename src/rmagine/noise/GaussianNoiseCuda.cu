#include "rmagine/noise/GaussianNoiseCuda.hpp"

#include <rmagine/util/cuda/random.cuh>

namespace rmagine
{


__global__ void kernel_gaussian_noise_apply(
    float* data,
    curandState* states,
    unsigned int N, 
    float mean, 
    float stddev)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        curandState state = states[idx];
        data[idx] += curand_normal(&state) * stddev + mean;
        // update state
        states[idx] = state;
    }
}

GaussianNoiseCuda::GaussianNoiseCuda(
    float mean, 
    float stddev, 
    NoiseCuda::Options options)
:NoiseCuda(options)
,m_mean(mean)
,m_stddev(stddev)
{

}

void GaussianNoiseCuda::apply(MemoryView<float, VRAM_CUDA>& ranges)
{
    updateStates(ranges);

    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (ranges.size() + blockSize - 1) / blockSize;

    kernel_gaussian_noise_apply<<<gridSize, blockSize>>>(ranges.raw(), m_states.raw(), ranges.size(), m_mean, m_stddev);
}

} // namespace rmagine