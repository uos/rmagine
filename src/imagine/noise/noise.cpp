#include "imagine/noise/noise.h"
#include <random>

namespace imagine 
{

GaussianNoise::GaussianNoise(float mean, float stddev)
:m_mean(mean)
,m_stddev(stddev)
{

}

template<>
void GaussianNoise::apply<RAM>(Memory<float, RAM>& ranges) const
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::default_random_engine gen;
    std::normal_distribution<float> distr(m_mean, m_stddev);

    for(size_t i=0; i<ranges.size(); i++)
    {
        ranges[i] += distr(gen);
    }
}

} // namespace imagine