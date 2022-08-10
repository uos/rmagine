#ifndef RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP
#define RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP

#include <rmagine/types/MemoryCuda.hpp>
#include <memory>
#include <utility>

#include "NoiseCuda.hpp"


namespace rmagine
{

class UniformDustNoiseCuda : public NoiseCuda {
public:
    UniformDustNoiseCuda(
        float hit_probability, 
        float return_probability, 
        NoiseCuda::Options options = {false, 0, true, 42});

    void apply(MemoryView<float, VRAM_CUDA>& ranges);
private:
    float m_hit_probability = 0.0;
    float m_return_probability = 1.0;
};

using UniformDustNoiseCudaPtr = std::shared_ptr<UniformDustNoiseCuda>;

} // namespace rmagine

#endif // RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP