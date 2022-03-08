#ifndef RMAGINE_OPTIX_CONTEXT_HPP
#define RMAGINE_OPTIX_CONTEXT_HPP

#include <rmagine/util/cuda/CudaContext.hpp>
#include <optix.h>
#include <optix_types.h>

#include <memory>

namespace rmagine
{

class OptixContext;

using OptixContextPtr = std::shared_ptr<OptixContext>;


class OptixContext {
public:
    OptixContext(CudaContextPtr cuda_context);

    ~OptixContext();

    static OptixContextPtr create(int device = 0)
    {
        CudaContextPtr cuda_ctx(new CudaContext(0));
        return std::make_shared<OptixContext>(cuda_ctx);
    }

    CudaContextPtr getCudaContext();

    OptixDeviceContext ref();


private:
    void init(CudaContextPtr cuda_context);

    CudaContextPtr m_cuda_context;
    
    OptixDeviceContext m_optix_context = nullptr;
};



static bool g_optix_initialized = false;

} // namespace rmagine

#endif // RMAGINE_OPTIX_CONTEXT_HPP