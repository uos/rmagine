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
    SVDCuda(CudaStreamPtr stream);

    ~SVDCuda();

    void calcUV(
        const MemoryView<Matrix3x3, VRAM_CUDA>& As,
        MemoryView<Matrix3x3, VRAM_CUDA>& Us,
        MemoryView<Matrix3x3, VRAM_CUDA>& Vs
    ) const;

    void calcUSV(const MemoryView<Matrix3x3, VRAM_CUDA>& As,
        MemoryView<Matrix3x3, VRAM_CUDA>& Us,
        MemoryView<Vector, VRAM_CUDA>& Ss,
        MemoryView<Matrix3x3, VRAM_CUDA>& Vs) const;

private:
    // global parameters
    CudaStreamPtr       m_stream;
    cusolverDnHandle_t  cusolverH = NULL;
    gesvdjInfo_t        gesvdj_params = NULL;
};

using SVDCudaPtr = std::shared_ptr<SVDCuda>;

} // namespace rmagine

#endif // RMAGINE_MATH_SVD_CUDA_HPP