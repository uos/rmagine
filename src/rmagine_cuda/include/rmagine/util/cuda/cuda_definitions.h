#ifndef RMAGINE_UTIL_CUDA_DEFINITIONS_H
#define RMAGINE_UTIL_CUDA_DEFINITIONS_H

#include <memory>

namespace rmagine
{

class CudaContext;
class CudaStream;

using CudaContextPtr = std::shared_ptr<CudaContext>;
using CudaStreamPtr = std::shared_ptr<CudaStream>;

using CudaContextWPtr = std::weak_ptr<CudaContext>;
using CudaStreamWPtr = std::weak_ptr<CudaStream>;

} // namespace rmagine

#endif // RMAGINE_UTIL_CUDA_DEFINITIONS_H
