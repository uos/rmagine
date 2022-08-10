#ifndef RMAGINE_NOISE_GAUSSIAN_NOISE_HPP
#define RMAGINE_NOISE_GAUSSIAN_NOISE_HPP

#include "Noise.hpp"
#include <random>

namespace rmagine
{

class GaussianNoise : public Noise 
{
public:
    GaussianNoise(
        float mean, 
        float stddev, 
        Noise::Options opt = {});

    void apply(MemoryView<float, RAM>& ranges);

private:
    float m_mean;
    float m_stddev;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::default_random_engine gen;
    std::normal_distribution<float> m_distr;

};

using GaussianNoisePtr = std::shared_ptr<GaussianNoise>;

} // namespace rmagine

#endif // RMAGINE_NOISE_GAUSSIAN_NOISE_HPP