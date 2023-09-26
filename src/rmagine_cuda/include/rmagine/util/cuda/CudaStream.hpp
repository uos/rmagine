#ifndef RMAGINE_UTIL_CUDA_STREAM_HPP
#define RMAGINE_UTIL_CUDA_STREAM_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>

#include <rmagine/util/cuda/cuda_definitions.h>
#include "CudaContext.hpp"

namespace rmagine {

class CudaStream : std::enable_shared_from_this<CudaStream> 
{
public:
    CudaStream(CudaContextPtr ctx = cuda_current_context());
    CudaStream(unsigned int flags, CudaContextPtr ctx = cuda_current_context());
    ~CudaStream();

    cudaStream_t handle() const;
    CudaContextPtr context() const;
    void synchronize();

private:
    cudaStream_t m_stream = NULL;
    // weak connection to context
    CudaContextPtr m_ctx;
};

} // namespace rmagine

#endif // RMAGINE_UTIL_CUDA_STREAM_HPP