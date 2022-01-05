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
 * OptixUtil.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef IMAGINE_OPTIX_UTIL_HPP
#define IMAGINE_OPTIX_UTIL_HPP

#include <optix.h>
#include <string>


#if OPTIX_VERSION < 70300

#pragma message("Using Old Optix none-inline functions")

#ifdef __cplusplus
extern "C" {
#endif

// Optix fix for not inlining their code
// - complete code is in header files
// - can only included once. otherwise crash
// - fix: 
// -- provide function signatures here
// -- load header of optix (actual code) into OptixUtil.cpp

OptixResult optixUtilAccumulateStackSizes( 
        OptixProgramGroup programGroup, OptixStackSizes* stackSizes );

OptixResult optixUtilComputeStackSizes( 
        const OptixStackSizes* stackSizes,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepth,
        unsigned int*          directCallableStackSizeFromTraversal,
        unsigned int*          directCallableStackSizeFromState,
        unsigned int*          continuationStackSize );

OptixResult optixUtilComputeStackSizesDCSplit( const OptixStackSizes* stackSizes,
                                               unsigned int           dssDCFromTraversal,
                                               unsigned int           dssDCFromState,
                                               unsigned int           maxTraceDepth,
                                               unsigned int           maxCCDepth,
                                               unsigned int           maxDCDepthFromTraversal,
                                               unsigned int           maxDCDepthFromState,
                                               unsigned int*          directCallableStackSizeFromTraversal,
                                               unsigned int*          directCallableStackSizeFromState,
                                               unsigned int*          continuationStackSize );

OptixResult optixUtilComputeStackSizesCssCCTree( const OptixStackSizes* stackSizes,
                                                 unsigned int           cssCCTree,
                                                 unsigned int           maxTraceDepth,
                                                 unsigned int           maxDCDepth,
                                                 unsigned int*          directCallableStackSizeFromTraversal,
                                                 unsigned int*          directCallableStackSizeFromState,
                                                 unsigned int*          continuationStackSize );

OptixResult optixUtilComputeStackSizesSimplePathTracer( OptixProgramGroup        programGroupRG,
                                                        OptixProgramGroup        programGroupMS1,
                                                        const OptixProgramGroup* programGroupCH1,
                                                        unsigned int             programGroupCH1Count,
                                                        OptixProgramGroup        programGroupMS2,
                                                        const OptixProgramGroup* programGroupCH2,
                                                        unsigned int             programGroupCH2Count,
                                                        unsigned int*            directCallableStackSizeFromTraversal,
                                                        unsigned int*            directCallableStackSizeFromState,
                                                        unsigned int*            continuationStackSize );

#ifdef __cplusplus
} // extern "C"
#endif

#else

#pragma message("Using New Optix inline functions")
// from version 70300 on they inlined their functions
#include <optix_stack_size.h>

#endif // OPTIX_VERSION < 70300

namespace imagine {

/**
 * @brief Load PTX Program as string
 * 
 * @param program_name 
 * @return std::string 
 */
std::string loadProgramPtx(const std::string& program_name);



} // namespace imagine

#endif // IMAGINE_OPTIX_UTIL_HPP