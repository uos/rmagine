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
    float stddev)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        curandState state;
        curand_init(42, idx, 0, &state);
        data[idx] += curand_normal(&state) * stddev + mean;
    }
}

template<>
void GaussianNoise::apply<VRAM_CUDA>(Memory<float, VRAM_CUDA>& ranges) const
{
    kernel_gaussian_noise<<<ranges.size(), 1>>>(ranges.raw(), ranges.size(), m_mean, m_stddev);
}

} // namespace rmagine