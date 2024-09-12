/**
 * Copyright (c) 2021, University Osnabrück
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

/*
 * CudaDebug.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_UTIL_CUDA_DEBUG_HPP
#define RMAGINE_UTIL_CUDA_DEBUG_HPP

#include <stdio.h>
#include <sstream>
#include <rmagine/util/exceptions.h>
#include <cuda_runtime.h>

#define RM_CUDA_CHECK(call)    \
{                 \
   cudaError_t code = call; \
   cudaAssert(code, __FILE__, __PRETTY_FUNCTION__, __LINE__);  \
}

void cudaAssert(
   cudaError_t code, 
   const char* file, 
   const char* func,
   int line);

#ifdef NDEBUG
    #define RM_CUDA_DEBUG() 
#else  // NDEBUG
    #define RM_CUDA_DEBUG() \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess)  \
    { \
        printf("Error: %s\n", cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    }
#endif // defined NDEBUG

#endif // RMAGINE_UTIL_CUDA_DEBUG_HPP