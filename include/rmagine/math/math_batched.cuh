#ifndef RMAGINE_MATH_MATH_BATCHED_CUH
#define RMAGINE_MATH_MATH_BATCHED_CUH

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>
#include "math.cuh"


namespace rmagine 
{

//////////
// #sumBatched
void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    Memory<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const Memory<Matrix3x3, VRAM_CUDA>& data,
    Memory<Matrix3x3, VRAM_CUDA>& sums);

Memory<Matrix3x3, VRAM_CUDA> sumBatched(
    const Memory<Matrix3x3, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const Memory<float, VRAM_CUDA>& data,
    Memory<float, VRAM_CUDA>& sums);

Memory<float, VRAM_CUDA> sumBatched(
    const Memory<float, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const Memory<unsigned int, VRAM_CUDA>& data,
    Memory<unsigned int, VRAM_CUDA>& sums);

Memory<unsigned int, VRAM_CUDA> sumBatched(
    const Memory<unsigned int, VRAM_CUDA>& data,
    size_t batchSize);

//////////
// #sumBatched masked
void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<bool, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<bool, VRAM_CUDA>& mask,
    size_t batchSize);

void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<unsigned int, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<unsigned int, VRAM_CUDA>& mask,
    size_t batchSize);

void sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<uint8_t, VRAM_CUDA>& mask,
    Memory<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const Memory<Vector, VRAM_CUDA>& data,
    const Memory<uint8_t, VRAM_CUDA>& mask,
    size_t batchSize);

////////
// #covBatched
void covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& covs);

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    unsigned int batchSize);

void covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& corr,
    const Memory<unsigned int, VRAM_CUDA>& ncorr,
    Memory<Matrix3x3, VRAM_CUDA>& covs);

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const Memory<Vector, VRAM_CUDA>& m1, 
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& corr,
    const Memory<unsigned int, VRAM_CUDA>& ncorr,
    unsigned int batchSize);

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_BATCHED_CUH