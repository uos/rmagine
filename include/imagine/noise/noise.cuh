#ifndef IMAGINE_NOISE_CUDA_H
#define IMAGINE_NOISE_CUDA_H

#include "noise.h"
#include <imagine/types/MemoryCuda.hpp>

namespace imagine {

template<>
void GaussianNoise::apply<RAM_CUDA>(Memory<float, RAM_CUDA>& ranges) const;

/// VRAM_CUDA
template<>
void GaussianNoise::apply<VRAM_CUDA>(Memory<float, VRAM_CUDA>& ranges) const;

} // namespace imagine 

#endif // IMAGINE_NOISE_CUDA_H