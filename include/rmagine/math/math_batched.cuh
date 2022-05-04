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
    const MemoryView<Vector, VRAM_CUDA>& data,
    MemoryView<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const MemoryView<Matrix3x3, VRAM_CUDA>& data,
    MemoryView<Matrix3x3, VRAM_CUDA>& sums);

Memory<Matrix3x3, VRAM_CUDA> sumBatched(
    const MemoryView<Matrix3x3, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const MemoryView<float, VRAM_CUDA>& data,
    MemoryView<float, VRAM_CUDA>& sums);

Memory<float, VRAM_CUDA> sumBatched(
    const MemoryView<float, VRAM_CUDA>& data,
    size_t batchSize);

void sumBatched(
    const MemoryView<unsigned int, VRAM_CUDA>& data,
    MemoryView<unsigned int, VRAM_CUDA>& sums);

Memory<unsigned int, VRAM_CUDA> sumBatched(
    const MemoryView<unsigned int, VRAM_CUDA>& data,
    size_t batchSize);

//////////
// #sumBatched masked
void sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<bool, VRAM_CUDA>& mask,
    MemoryView<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<bool, VRAM_CUDA>& mask,
    size_t batchSize);

void sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<unsigned int, VRAM_CUDA>& mask,
    MemoryView<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<unsigned int, VRAM_CUDA>& mask,
    size_t batchSize);

void sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<uint8_t, VRAM_CUDA>& mask,
    MemoryView<Vector, VRAM_CUDA>& sums);

Memory<Vector, VRAM_CUDA> sumBatched(
    const MemoryView<Vector, VRAM_CUDA>& data,
    const MemoryView<uint8_t, VRAM_CUDA>& mask,
    size_t batchSize);

////////
// #covBatched
void covBatched(
    const MemoryView<Vector, VRAM_CUDA>& m1, 
    const MemoryView<Vector, VRAM_CUDA>& m2,
    MemoryView<Matrix3x3, VRAM_CUDA>& covs);

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const MemoryView<Vector, VRAM_CUDA>& m1, 
    const MemoryView<Vector, VRAM_CUDA>& m2,
    unsigned int batchSize);

void covBatched(
    const MemoryView<Vector, VRAM_CUDA>& m1, 
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& corr,
    const MemoryView<unsigned int, VRAM_CUDA>& ncorr,
    MemoryView<Matrix3x3, VRAM_CUDA>& covs);

Memory<Matrix3x3, VRAM_CUDA> covBatched(
    const MemoryView<Vector, VRAM_CUDA>& m1, 
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& corr,
    const MemoryView<unsigned int, VRAM_CUDA>& ncorr,
    unsigned int batchSize);

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_BATCHED_CUH