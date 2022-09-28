#ifndef RMAGINE_NOISE_NOISE_HPP
#define RMAGINE_NOISE_NOISE_HPP

#include <rmagine/types/Memory.hpp>
#include <memory>

namespace rmagine
{

class Noise 
{
public:
    struct Options {
        unsigned int seed = 42;
        float max_range = 10000.0;
    };

    Noise(Options options = {42, 10000.0});

    virtual void apply(MemoryView<float, RAM>& ranges) = 0;
protected:
    Options m_options;
};

using NoisePtr = std::shared_ptr<Noise>;

} // namespace rmagine

#endif // RMAGINE_NOISE_NOISE_HPP