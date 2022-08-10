#ifndef RMAGINE_NOISE_GAUSSIAN_NOISE_CUDA_HPP
#define RMAGINE_NOISE_GAUSSIAN_NOISE_CUDA_HPP

#include <rmagine/types/MemoryCuda.hpp>
#include <memory>
#include <utility>

#include "NoiseCuda.hpp"

namespace rmagine
{

class GaussianNoiseCuda : public NoiseCuda {
public:
    GaussianNoiseCuda(
        float mean, 
        float stddev, 
        NoiseCuda::Options options = {false, 0, true, 42});

    void apply(MemoryView<float, VRAM_CUDA>& ranges);
private:
    float m_mean = 0.0;
    float m_stddev = 0.0;
};

using GaussianNoiseCudaPtr = std::shared_ptr<GaussianNoiseCuda>;

} // namespace rmagine

#endif // RMAGINE_NOISE_GAUSSIAN_NOISE_CUDA_HPP