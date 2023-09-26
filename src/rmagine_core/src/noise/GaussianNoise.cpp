#include "rmagine/noise/GaussianNoise.hpp"

#include <random>

namespace rmagine
{

GaussianNoise::GaussianNoise(
    float mean, 
    float stddev, 
    Noise::Options options)
:Noise(options)
,m_mean(mean)
,m_stddev(stddev)
,m_distr(mean, stddev)
{

}

void GaussianNoise::apply(MemoryView<float, RAM>& ranges)
{
    const float max_range = m_options.max_range;
    for(size_t i=0; i<ranges.size(); i++)
    {
        const float range = ranges[i];
        if(range <= max_range)
        {
            ranges[i] += m_distr(gen);
        }
    }
}

} // namespace rmagine