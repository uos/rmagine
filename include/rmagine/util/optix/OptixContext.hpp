#ifndef RMAGINE_OPTIX_CONTEXT_HPP
#define RMAGINE_OPTIX_CONTEXT_HPP

#include <rmagine/util/cuda/CudaContext.hpp>
#include <memory>

// Optix Forward declarations
struct OptixDeviceContext_t;
typedef struct OptixDeviceContext_t* OptixDeviceContext;

namespace rmagine
{

class OptixContext;

using OptixContextPtr = std::shared_ptr<OptixContext>;

bool optix_initialized();
void optix_initialize();

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

OptixContextPtr optix_default_context();
// void reset_optix_default_context();

} // namespace rmagine

#endif // RMAGINE_OPTIX_CONTEXT_HPP