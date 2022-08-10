#include "rmagine/noise/noise.cuh"
#include <random>
#include <curand.h>
#include <curand_kernel.h>

namespace rmagine 
{

template<>
void GaussianNoise::apply<RAM_CUDA>(Memory<float, RAM_CUDA>& ranges) const
{
    std::default_random_engine gen;
    std::normal_distribution<float> distr(m_mean, m_stddev);

    #pragma omp parallel for
    for(size_t i=0; i<ranges.size(); i++)
    {
        ranges[i] += distr(gen);
    }
}

__global__ void kernel_gaussian_noise(
    float* data, 
    unsigned int N, 
    float mean, 
    float stddev,
    unsigned int seed = 42)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] += curand_normal(&state) * stddev + mean;
    }
}

// __global__ void kernel_curand_init(
//     curandState* states, 
//     unsigned int N,
//     unsigned int seed = 42)
// {
//     unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if(idx < N)
//     {
//         curand_init(seed, idx, 0, &states[idx]);
//     }
// }

__global__ void kernel_gaussian_noise(
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
        data[idx] += curand_normal(&state) * stddev + mean;;
        // update state
        states[idx] = state;
    }
}

template<>
void GaussianNoise::apply<VRAM_CUDA>(Memory<float, VRAM_CUDA>& ranges) const
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (ranges.size() + blockSize - 1) / blockSize;

    kernel_gaussian_noise<<<gridSize, blockSize>>>(ranges.raw(), ranges.size(), m_mean, m_stddev);
}

} // namespace rmagine