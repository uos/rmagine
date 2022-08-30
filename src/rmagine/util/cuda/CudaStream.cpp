#include "rmagine/util/cuda/CudaStream.hpp"

#include "rmagine/util/cuda/CudaDebug.hpp"
#include "rmagine/util/cuda/CudaContext.hpp"

namespace rmagine
{

CudaStream::CudaStream(CudaContextPtr ctx)
:m_ctx(ctx)
{
    CUcontext old;
    cuCtxGetCurrent(&old);

    ctx->use();
    RM_CUDA_CHECK( cudaStreamCreate(&m_stream) );

    // restore old
    cuCtxSetCurrent(old);
}

CudaStream::CudaStream(unsigned int flags, CudaContextPtr ctx)
:m_ctx(ctx)
{
    CUcontext old;
    cuCtxGetCurrent(&old);

    ctx->use();
    RM_CUDA_CHECK( cudaStreamCreateWithFlags(&m_stream, flags) );

    // restore old
    cuCtxSetCurrent(old);
}

CudaStream::~CudaStream()
{
    cudaStreamDestroy(m_stream);
}

cudaStream_t CudaStream::handle() const
{
    return m_stream;
}

CudaContextPtr CudaStream::context() const
{
    return m_ctx;
}

void CudaStream::synchronize()
{
    RM_CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

} // namespace rmagine