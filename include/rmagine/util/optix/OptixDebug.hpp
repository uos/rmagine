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
 * OptixDebug.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_UTIL_OPTIX_DEBUG_HPP
#define RMAGINE_UTIL_OPTIX_DEBUG_HPP

#include <cuda_runtime.h>

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <rmagine/util/exceptions.h>

#include <rmagine/util/cuda/CudaDebug.hpp>


#define RM_OPTIX_CHECK( call )                                                              \
{                                                                                           \
    OptixResult res = call;                                                                 \
    if( res != OPTIX_SUCCESS )                                                              \
    {                                                                                       \
        std::stringstream ss;                                                               \
        ss << "Optix call '" << #call << "' failed."                                        \
            << " Error Name: " << optixGetErrorName(res)                                    \
            << ", Error Msg: " << optixGetErrorString(res);                                 \
        throw rmagine::OptixException(ss.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__);   \
    }                                                                                       \
}


#define RM_OPTIX_CHECK_LOG( call )                                                \
{                                                                          \
    OptixResult res = call;                                                \
    const size_t sizeof_log_returned = sizeof_log;                         \
    sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
    if( res != OPTIX_SUCCESS )                                             \
    {                                                                      \
        std::stringstream ss;                                              \
        ss << "Optix call '" << #call << "' failed. "               \
            << " Error Name: " << optixGetErrorName(res)                                    \
            << ", Error Msg: " << optixGetErrorString(res)       \
            << "\nLog:\n" << log                               \
            << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
            << "\n";                                                        \
        throw rmagine::OptixException( ss.str(), __FILE__, __PRETTY_FUNCTION__, __LINE__ ); \
    }                                                                      \
}


#endif // RMAGINE_UTIL_OPTIX_DEBUG_HPP