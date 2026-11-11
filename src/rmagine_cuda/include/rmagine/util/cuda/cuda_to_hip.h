/*
 * cuda_to_hip.h
 *
 * Single compatibility shim that lets the rmagine CUDA backend build with the
 * ROCm/HIP toolchain (USE_HIP) while keeping the NVIDIA path byte-identical.
 *
 * Under HIP it pulls in the HIP runtime and driver headers and maps the small
 * set of cuda and cuCtx symbols rmagine_cuda actually uses onto their hip
 * spellings. The sources stay in CUDA spelling and are marked LANGUAGE HIP in
 * CMake. Outside HIP it just includes the real CUDA headers, so a CUDA build
 * sees no change.
 *
 * This header is included in place of the cuda_runtime and cuda headers at
 * every rmagine_cuda translation unit. The RNG mapping lives next door in
 * curand_to_hiprand.h so that a translation unit that only touches the runtime
 * does not drag a device-side RNG header in with it.
 */
#ifndef RMAGINE_UTIL_CUDA_TO_HIP_H
#define RMAGINE_UTIL_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// --- runtime API ---
#define cudaError_t                 hipError_t
#define cudaSuccess                 hipSuccess
#define cudaGetLastError            hipGetLastError
#define cudaGetErrorString          hipGetErrorString
#define cudaGetErrorName            hipGetErrorName
#define cudaGetDeviceCount          hipGetDeviceCount
#define cudaGetDeviceProperties     hipGetDeviceProperties
#define cudaDeviceProp              hipDeviceProp_t
#define cudaRuntimeGetVersion       hipRuntimeGetVersion
#define cudaDriverGetVersion        hipDriverGetVersion

#define cudaMalloc                  hipMalloc
#define cudaFree                    hipFree
#define cudaMallocHost              hipHostMalloc
#define cudaFreeHost                hipHostFree
#define cudaMallocManaged           hipMallocManaged
#define cudaMemcpy                  hipMemcpy
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemset                  hipMemset
#define cudaMemcpyHostToHost        hipMemcpyHostToHost
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice

#define cudaStream_t                hipStream_t
#define cudaStreamCreate            hipStreamCreate
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamDestroy           hipStreamDestroy
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaDeviceSynchronize       hipDeviceSynchronize

// --- driver API (context management) ---
#define CUcontext                   hipCtx_t
#define CUresult                    hipError_t
#define CUDA_SUCCESS                hipSuccess
#define cuInit                      hipInit
#define cuCtxCreate                 hipCtxCreate
#define cuCtxDestroy                hipCtxDestroy
#define cuCtxGetCurrent             hipCtxGetCurrent
#define cuCtxSetCurrent             hipCtxSetCurrent
#define cuCtxPushCurrent            hipCtxPushCurrent
#define cuCtxGetDevice              hipCtxGetDevice
#define cuCtxSynchronize            hipCtxSynchronize

#else // CUDA

#include <cuda_runtime.h>
#include <cuda.h>

#endif // USE_HIP

#endif // RMAGINE_UTIL_CUDA_TO_HIP_H
