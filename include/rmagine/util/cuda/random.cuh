#ifndef RMAGINE_UTIL_CUDA_RANDOM_CUH
#define RMAGINE_UTIL_CUDA_RANDOM_CUH

#include <rmagine/types/MemoryCuda.hpp>
#include <curand.h>
#include <curand_kernel.h>

namespace rmagine
{

void random_init(
    MemoryView<curandState, VRAM_CUDA>& states,
    unsigned int seed = 42
);

} // namespace rmagine

#endif // RMAGINE_UTIL_CUDA_RANDOM_CUH