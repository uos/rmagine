
#include <imagine/types/Memory.hpp>

namespace imagine
{

// template<typename MemT>
// class Noise {
// public:
//     virtual void apply(Memory<float, MemT>& meas) = 0;
// };

template<typename MemT>
class GaussianNoise : public Noise<MemT> {
public:
    GaussianNoise(float mean, float sigma)
    :m_mean(mean)
    ,m_sigma(sigma)
    {

    }

    Memory<float, MemT> apply(const Memory<float, MemT>& ranges) const;

private:
    float m_mean;
    float m_sigma;
};

} // namespace imagine