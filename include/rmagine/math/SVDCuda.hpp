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
 * @brief SVD Solver for CUDA Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_SVD_CUDA_HPP
#define RMAGINE_MATH_SVD_CUDA_HPP

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/types.h>
#include <memory>
#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine {

class SVDCuda
{
public:
    SVDCuda(CudaContextPtr ctx = cuda_current_context());
    SVDCuda(CudaStreamPtr stream);

    ~SVDCuda();

    void calcUV(
        const MemoryView<Matrix3x3, VRAM_CUDA>& As,
        MemoryView<Matrix3x3, VRAM_CUDA>& Us,
        MemoryView<Matrix3x3, VRAM_CUDA>& Vs
    ) const;

    void calcUSV(const MemoryView<Matrix3x3, VRAM_CUDA>& As,
        MemoryView<Matrix3x3, VRAM_CUDA>& Us,
        MemoryView<Vector, VRAM_CUDA>& Ss,
        MemoryView<Matrix3x3, VRAM_CUDA>& Vs) const;

private:
    // global parameters
    CudaStreamPtr       m_stream;
    cusolverDnHandle_t  cusolverH = NULL;
    gesvdjInfo_t        gesvdj_params = NULL;
};

using SVDCudaPtr = std::shared_ptr<SVDCuda>;

} // namespace rmagine

#endif // RMAGINE_MATH_SVD_CUDA_HPP