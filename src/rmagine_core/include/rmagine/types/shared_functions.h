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
 * @brief Function Declerators to share function signatures between CPU and GPU code
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_TYPES_SHARED_FUNCTIONS_H
#define RMAGINE_TYPES_SHARED_FUNCTIONS_H

#ifdef __CUDA_ARCH__
#define RMAGINE_FUNCTION __host__ __device__
#define RMAGINE_INLINE_FUNCTION __inline__ __host__ __device__ 
#define RMAGINE_HOST_FUNCTION __host__
#define RMAGINE_INLINE_HOST_FUNCTION __inline__ __host__
#define RMAGINE_DEVICE_FUNCTION __device__
#define RMAGINE_INLINE_DEVICE_FUNCTION __inline__ __device__
#else
#define RMAGINE_FUNCTION
#define RMAGINE_INLINE_FUNCTION inline
#define RMAGINE_HOST_FUNCTION 
#define RMAGINE_INLINE_HOST_FUNCTION inline
#define RMAGINE_DEVICE_FUNCTION 
#define RMAGINE_INLINE_DEVICE_FUNCTION inline
#endif



#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define RMAGINE_API __attribute__ ((dllexport))
    #else
      #define RMAGINE_API __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define RMAGINE_API __attribute__ ((dllimport))
    #else
      #define RMAGINE_API __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define RMAGINE_API __attribute__ ((visibility ("default")))
    #define RMAGINE_HIDDEN  __attribute__ ((visibility ("hidden")))
  #else
    #define RMAGINE_API
    #define RMAGINE_HIDDEN
  #endif
#endif

#endif // RMAGINE_TYPES_SHARED_FUNCTIONS_H