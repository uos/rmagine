#ifndef IMAGINE_NOISE_H
#define IMAGINE_NOISE_H

#include <rmagine/types/Memory.hpp>

namespace rmagine
{

class GaussianNoise
{
public:
    GaussianNoise(float mean, float stddev);

    template<typename MemT>
    void apply(Memory<float, MemT>& ranges) const;

private:
    float m_mean;
    float m_stddev;
};

/// RAM
template<>
void GaussianNoise::apply<RAM>(Memory<float, RAM>& ranges) const;

} // namespace rmagine

#include "noise.tcc"

#endif // IMAGINE_NOISE_H