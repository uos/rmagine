#ifndef RMAGINE_OPTIX_CONTEXT_HPP
#define RMAGINE_OPTIX_CONTEXT_HPP

#include <rmagine/util/cuda/CudaContext.hpp>
#include <optix.h>
#include <optix_types.h>

#include <memory>

namespace rmagine
{

class OptixContext {
public:
    OptixContext();
    
    OptixContext(int device_id);

    OptixContext(CudaContextPtr cuda_context);

    ~OptixContext();

    CudaContextPtr getCudaContext();

    OptixDeviceContext ref();

private:
    void init(CudaContextPtr cuda_context);

    CudaContextPtr m_cuda_context;
    
    OptixDeviceContext m_optix_context = nullptr;
};

using OptixContextPtr = std::shared_ptr<OptixContext>;

static OptixContextPtr g_optix_context;
static bool g_optix_initialized = false;

} // namespace rmagine

#endif // RMAGINE_OPTIX_CONTEXT_HPP