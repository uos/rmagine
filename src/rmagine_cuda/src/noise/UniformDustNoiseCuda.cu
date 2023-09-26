#include "rmagine/noise/UniformDustNoiseCuda.hpp"

#include <rmagine/util/cuda/random.cuh>

namespace rmagine
{

__global__ void kernel_uniform_dust_noise_apply(
    float* data,
    curandState* states,
    unsigned int N, 
    float hit_probability,
    float return_probability,
    float max_range)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
    {
        curandState state = states[idx];
        float range = data[idx];

        if(range > max_range)
        {
            range = max_range;
        }

        // compute total probability
        // from hit probability per meter with actual range
        const float p_hit = 1.0 - powf(1.0 - hit_probability, range);
        const float p_hit_rand = curand_uniform(&state);

        if(p_hit_rand < p_hit)
        {
            const float new_range = curand_uniform(&state) * range;

            // calculate return probability dependend on new range
            const float p_return = powf(return_probability, new_range);

            const float p_return_rand = curand_uniform(&state);
            if(p_return_rand < p_return)
            {
                data[idx] = new_range;
            }
        }

        // update state
        states[idx] = state;
    }
}

UniformDustNoiseCuda::UniformDustNoiseCuda(
    float hit_probability,
    float return_probability, 
    NoiseCuda::Options options)
:NoiseCuda(options)
,m_hit_probability(hit_probability)
,m_return_probability(return_probability)
{
    
}

void UniformDustNoiseCuda::apply(MemoryView<float, VRAM_CUDA>& ranges)
{
    updateStates(ranges);

    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (ranges.size() + blockSize - 1) / blockSize;

    kernel_uniform_dust_noise_apply<<<gridSize, blockSize>>>(
        ranges.raw(), m_states.raw(), ranges.size(), 
        m_hit_probability, m_return_probability, m_options.max_range);
}

} // namespace rmagine