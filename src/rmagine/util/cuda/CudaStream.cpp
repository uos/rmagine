#include "rmagine/util/cuda/CudaStream.hpp"

namespace rmagine
{

CudaStream::CudaStream()
{
    cudaStreamCreate(&m_stream);
}

CudaStream::~CudaStream()
{
    cudaStreamDestroy(m_stream);
}

cudaStream_t CudaStream::handle()
{
    return m_stream;
}

} // namespace rmagine