/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Math Functions for batched CUDA Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_MATH_BATCHED_CUH
#define RMAGINE_MATH_MATH_BATCHED_CUH

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/memory_math.cuh>


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