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
 * @brief GPU Noise base class
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_NOISE_NOISE_CUDA_HPP
#define RMAGINE_NOISE_NOISE_CUDA_HPP

#include <rmagine/types/MemoryCuda.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include <utility>

namespace rmagine
{

class NoiseCuda {
public:

    struct Options {
        bool fixed_memory = false; // if you want to disable any dynamic memory management. estimated_memory_size must be set wisely
        unsigned int estimated_memory_size = 0; // estimate the number of elements for ranges. >= is goot estimate
        bool never_shrink_memory = true; // perfect performance but more memory consuming
        unsigned int seed = 42;
        float max_range = 10000.0;
    };

    NoiseCuda(
        Options options = {false, 0, true, 42, 10000.0});

    virtual void apply(MemoryView<float, VRAM_CUDA>& ranges) = 0;

protected:
    void updateStates(MemoryView<float, VRAM_CUDA>& ranges);

    Options m_options;
    Memory<curandState, VRAM_CUDA> m_states;
};

using NoiseCudaPtr = std::shared_ptr<NoiseCuda>;

} // namespace rmagine

#endif // RMAGINE_NOISE_NOISE_CUDA_HPP