#include "rmagine/util/optix/OptixContext.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix.h>
#include <optix_stubs.h>
#include <iomanip>

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

namespace rmagine {

OptixContext::OptixContext()
{
    if(!g_cuda_initialized)
    {
        cuInit(0);
        g_cuda_initialized = true;
    }

    if(!g_cuda_context)
    {
        g_cuda_context.reset(new CudaContext());
    }

    m_cuda_context = g_cuda_context;
    init(g_cuda_context);
}

OptixContext::OptixContext(int device_id)
{
    if(!g_cuda_initialized)
    {
        cuInit(0);
        g_cuda_initialized = true;
    }

    if(!g_cuda_context)
    {
        g_cuda_context.reset(new CudaContext(device_id));
    }

    m_cuda_context = g_cuda_context;
    init(g_cuda_context);
}

OptixContext::OptixContext(CudaContextPtr cuda_context)
:m_cuda_context(cuda_context)
{
    init(cuda_context);
}

OptixContext::~OptixContext()
{
    optixDeviceContextDestroy( m_optix_context );
}

CudaContextPtr OptixContext::getCudaContext()
{
    return m_cuda_context;
}

OptixDeviceContext OptixContext::ref()
{
    return m_optix_context;
}

void OptixContext::init(CudaContextPtr cuda_context)
{
    if(!g_optix_initialized)
    {

        std::stringstream optix_version_str;
        optix_version_str << OPTIX_VERSION / 10000 << "." << (OPTIX_VERSION % 10000) / 100 << "." << OPTIX_VERSION % 100;

        std::cout << "[OptixContext] Init Optix (" << optix_version_str.str() << ") context on latest CUDA context " << std::endl;

        OPTIX_CHECK( optixInit() );
        g_optix_initialized = true;
    }

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 3;

    OPTIX_CHECK( optixDeviceContextCreate( cuda_context->ref(), &options, &m_optix_context ) );
}

} // namespace rmagine