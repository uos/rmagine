/*
 * curand_to_hiprand.h
 *
 * RNG half of the compatibility shim that lets the rmagine CUDA backend build
 * with the ROCm/HIP toolchain (USE_HIP) while keeping the NVIDIA path
 * byte-identical.
 *
 * Under HIP it pulls in the hipRAND device header and maps the curand symbols
 * rmagine_cuda uses onto their hiprand spellings. Outside HIP it just includes
 * the real cuRAND headers.
 *
 * This is separate from cuda_to_hip.h because only the two headers that name
 * curandState in a declaration need it. The device-side RNG header is heavy and
 * is not meant for host translation units, so keep it out of the runtime shim
 * that every rmagine_cuda file includes.
 */
#ifndef RMAGINE_UTIL_CURAND_TO_HIPRAND_H
#define RMAGINE_UTIL_CURAND_TO_HIPRAND_H

#include <rmagine/util/cuda/cuda_to_hip.h>

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// rocrand_mtgp32.h calls printf in a host function without including <cstdio>
// itself, so include it first to keep this header usable from a plain C++
// translation unit whatever the surrounding include order is.
#include <cstdio>
#include <hiprand/hiprand_kernel.h>

#define curandState                 hiprandState
#define curand_init                 hiprand_init
#define curand_uniform              hiprand_uniform
#define curand_normal               hiprand_normal

#else // CUDA

#include <curand.h>
#include <curand_kernel.h>

#endif // USE_HIP

#endif // RMAGINE_UTIL_CURAND_TO_HIPRAND_H
