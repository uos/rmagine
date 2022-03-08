#ifndef RMAGINE_NOISE_CUDA_H
#define RMAGINE_NOISE_CUDA_H

#include "noise.h"
#include <rmagine/types/MemoryCuda.hpp>

namespace rmagine {

template<>
void GaussianNoise::apply<RAM_CUDA>(Memory<float, RAM_CUDA>& ranges) const;

/// VRAM_CUDA
template<>
void GaussianNoise::apply<VRAM_CUDA>(Memory<float, VRAM_CUDA>& ranges) const;

} // namespace rmagine 

#endif // RMAGINE_NOISE_CUDA_H