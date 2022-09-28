#ifndef RMAGINE_NOISE_NOISE_CUDA_HPP
#define RMAGINE_NOISE_NOISE_CUDA_HPP

#include <rmagine/types/MemoryCuda.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include <utility>

namespace rmagine
{


class NoiseCuda {
public:

    struct Options {
        bool fixed_memory = false; // if you want to disable any dynamic memory management. estimated_memory_size must be set wisely
        unsigned int estimated_memory_size = 0; // estimate the number of elements for ranges. >= is goot estimate
        bool never_shrink_memory = true; // perfect performance but more memory consuming
        unsigned int seed = 42;
        float max_range = 10000.0;
    };

    NoiseCuda(
        Options options = {false, 0, true, 42, 10000.0});

    virtual void apply(MemoryView<float, VRAM_CUDA>& ranges) = 0;

protected:
    void updateStates(MemoryView<float, VRAM_CUDA>& ranges);

    Options m_options;
    Memory<curandState, VRAM_CUDA> m_states;
};

using NoiseCudaPtr = std::shared_ptr<NoiseCuda>;

} // namespace rmagine

#endif // RMAGINE_NOISE_NOISE_CUDA_HPP