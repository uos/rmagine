#include "rmagine/noise/RelGaussianNoise.hpp"

#include <random>

namespace rmagine
{

RelGaussianNoise::RelGaussianNoise(
    float mean, 
    float stddev, 
    float range_exp,
    Noise::Options options)
:Noise(options)
,m_mean(mean)
,m_stddev(stddev)
,m_range_exp(range_exp)
,m_distr(0.0, 1.0)
{

}

void RelGaussianNoise::apply(MemoryView<float, RAM>& ranges)
{
    const float max_range = m_options.max_range;
    for(size_t i=0; i<ranges.size(); i++)
    {
        const float range = ranges[i];
        if(range <= max_range)
        {
            const float stddev_range = m_stddev * powf(range, m_range_exp);
            ranges[i] += m_distr(gen) * stddev_range + m_mean;
        }
    }
}

} // namespace rmagine