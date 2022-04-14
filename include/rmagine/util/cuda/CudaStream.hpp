#include <cuda_runtime.h>
#include <memory>

namespace rmagine {

class CudaStream {
public:
    CudaStream();
    ~CudaStream();

    cudaStream_t handle();
private:
    cudaStream_t m_stream = NULL;
};

using CudaStreamPtr = std::shared_ptr<CudaStream>;

} // namespace rmagine