#ifndef RMAGINE_NOISE_UNIFORM_DUST_NOISE_HPP
#define RMAGINE_NOISE_UNIFORM_DUST_NOISE_HPP

#include "Noise.hpp"
#include <random>

namespace rmagine
{

class UniformDustNoise : public Noise 
{
public:
    UniformDustNoise(
        float hit_prob, 
        float ret_prob, 
        Noise::Options opt = {});

    void apply(MemoryView<float, RAM>& ranges);

private:
    float m_hit_prob;
    float m_ret_prob;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::default_random_engine gen;

    std::uniform_real_distribution<float> m_distr;

};

using UniformDustNoisePtr = std::shared_ptr<UniformDustNoise>;

} // namespace rmagine

#endif // RMAGINE_NOISE_UNIFORM_DUST_NOISE_HPP