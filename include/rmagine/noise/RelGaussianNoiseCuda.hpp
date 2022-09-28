#ifndef RMAGINE_NOISE_REL_GAUSSIAN_NOISE_CUDA_HPP
#define RMAGINE_NOISE_REL_GAUSSIAN_NOISE_CUDA_HPP


#include "NoiseCuda.hpp"

namespace rmagine
{

class RelGaussianNoiseCuda : public NoiseCuda
{
public:
    /**
     * @brief Construct a new Rel Gaussian Noise Cuda object
     * 
     * stddev_range = stddev * range^range_exp
     * 
     * @param mean 
     * @param stddev 
     * @param range_exp 
     * @param options 
     */
    RelGaussianNoiseCuda(
        float mean, 
        float stddev,
        float range_exp,
        NoiseCuda::Options options = {false, 0, true, 42});

    void apply(MemoryView<float, VRAM_CUDA>& ranges);
private:
    float m_mean = 0.0;
    float m_stddev = 0.0;
    float m_range_exp = 1.0;
};

using RelGaussianNoiseCudaPtr = std::shared_ptr<RelGaussianNoiseCuda>;

} // namespace rmagine



#endif // RMAGINE_NOISE_REL_GAUSSIAN_NOISE_CUDA_HPP