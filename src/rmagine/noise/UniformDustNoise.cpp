#include "rmagine/noise/UniformDustNoise.hpp"

#include <random>

namespace rmagine
{

UniformDustNoise::UniformDustNoise(
    float hit_prob, 
    float ret_prob, 
    Noise::Options options)
:Noise(options)
,m_hit_prob(hit_prob)
,m_ret_prob(ret_prob)
,m_distr(0.0, 1.0)
{

}

void UniformDustNoise::apply(MemoryView<float, RAM>& ranges)
{
    for(size_t i=0; i<ranges.size(); i++)
    {
        const float range = ranges[i];

        // compute total probability
        // from hit probability per meter with actual range
        const float p_hit = 1.0 - powf(1.0 - m_hit_prob, range);
        const float p_hit_rand = m_distr(gen);

        if(p_hit_rand < p_hit)
        {
            const float new_range = m_distr(gen) * range;
            const float p_return = powf(m_ret_prob, new_range);

            const float p_return_rand = m_distr(gen);

            if(p_return_rand < p_return)
            {
                ranges[i] = new_range;
            }
        }
    }
}

} // namespace rmagine