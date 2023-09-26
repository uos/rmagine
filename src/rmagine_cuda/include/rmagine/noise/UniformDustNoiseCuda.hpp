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
 * @brief Uniform Dust Noise on GPU Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP
#define RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP

#include <rmagine/types/MemoryCuda.hpp>
#include <memory>
#include <utility>

#include "NoiseCuda.hpp"


namespace rmagine
{

class UniformDustNoiseCuda : public NoiseCuda {
public:
    UniformDustNoiseCuda(
        float hit_probability, 
        float return_probability, 
        NoiseCuda::Options options = {false, 0, true, 42});

    void apply(MemoryView<float, VRAM_CUDA>& ranges);
private:
    float m_hit_probability = 0.0;
    float m_return_probability = 1.0;
};

using UniformDustNoiseCudaPtr = std::shared_ptr<UniformDustNoiseCuda>;

} // namespace rmagine

#endif // RMAGINE_NOISE_UNIFORM_DUST_NOISE_CUDA_HPP