#ifndef RMAGINE_NOISE_REL_GAUSSIAN_NOISE_HPP
#define RMAGINE_NOISE_REL_GAUSSIAN_NOISE_HPP

#include "Noise.hpp"
#include <random>

namespace rmagine
{

class RelGaussianNoise : public Noise 
{
public:
    RelGaussianNoise(
        float mean,
        float stddev,
        float range_exp,
        Noise::Options opt = {});

    void apply(MemoryView<float, RAM>& ranges);
private:
    float m_mean;
    float m_stddev;
    float m_range_exp;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::default_random_engine gen;
    std::normal_distribution<float> m_distr;
};

using RelGaussianNoisePtr = std::shared_ptr<RelGaussianNoise>;

} // namespace rmagine

#endif // RMAGINE_NOISE_REL_GAUSSIAN_NOISE_HPP