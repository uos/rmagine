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
 * CudaContext.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_UTIL_CUDA_CONTEXT_HPP
#define RMAGINE_UTIL_CUDA_CONTEXT_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "CudaHelper.hpp"

#include "cuda_definitions.h"

namespace rmagine {

static void printCudaInfo()
{
    int driver;
    cudaDriverGetVersion(&driver);
    int cuda_version;
    cudaRuntimeGetVersion(&cuda_version);

    std::stringstream driver_version_str, cuda_version_str;
    driver_version_str << driver / 1000 << "." << (driver % 1000) / 10 << "." << driver % 10;
    cuda_version_str << cuda_version / 1000 << "." << (cuda_version % 1000) / 10 << "." << cuda_version % 10;

    std::cout << "[RMagine] Latest CUDA for driver: " << driver_version_str.str() << ". Current CUDA version: " << cuda_version_str.str() << std::endl;
}

bool cuda_initialized();
void cuda_initialize();

class CudaContext : std::enable_shared_from_this<CudaContext>
{
public:
    CudaContext(int device_id = 0);
    CudaContext(CUcontext ctx);
    ~CudaContext();

    int getDeviceId() const;
    cudaDeviceProp getDeviceInfo() const;
    void use();
    void enqueue();
    bool isActive() const;

    CudaStreamPtr createStream(unsigned int flags = 0) const;

    // Only 4 and 8 Bytes are supported yet
    void setSharedMemBankSize(unsigned int bytes);
    
    unsigned int getSharedMemBankSize() const;

    void synchronize();

    CUcontext ref();

    friend std::ostream& operator<<(std::ostream& os, const CudaContext& dt);

private:
    CUcontext m_context;
};

using CudaContextPtr = std::shared_ptr<CudaContext>;

CudaContextPtr cuda_current_context();

} // namespace rmagine

#endif // RMAGINE_UTIL_CUDA_CONTEXT_HPP