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
    SVDCuda();
    SVDCuda(cudaStream_t stream);

    ~SVDCuda();

    void calcUV(
        const Memory<Matrix3x3, VRAM_CUDA>& As,
        Memory<Matrix3x3, VRAM_CUDA>& Us,
        Memory<Matrix3x3, VRAM_CUDA>& Vs
    ) const;

    void calcUSV(const Memory<Matrix3x3, VRAM_CUDA>& As,
        Memory<Matrix3x3, VRAM_CUDA>& Us,
        Memory<Vector, VRAM_CUDA>& Ss,
        Memory<Matrix3x3, VRAM_CUDA>& Vs) const;

private:
    // global parameters
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
};

using SVDCudaPtr = std::shared_ptr<SVDCuda>;

} // namespace rmagine

#endif // RMAGINE_MATH_SVD_CUDA_HPP