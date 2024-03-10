#include "rmagine/util/optix/OptixContext.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix.h>
#include <optix_stubs.h>
#include <iomanip>
#include <map>
#include <utility>


std::map<unsigned int, unsigned int> optix_driver_map = {
    {70200, 45671},
    {70300, 46584},
    {70400, 49589},
    {70500, 49589}
};

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

namespace rmagine {

bool optix_initialized_ = false;

bool optix_initialized()
{
    return optix_initialized_;
}

void optix_initialize()
{
    std::stringstream optix_version_str;
    optix_version_str << OPTIX_VERSION / 10000 << "." << (OPTIX_VERSION % 10000) / 100 << "." << OPTIX_VERSION % 100;

    auto driver_it = optix_driver_map.upper_bound(OPTIX_VERSION);
    unsigned int required_driver_version = 0;
    if(driver_it == optix_driver_map.begin())
    {
        required_driver_version = 45671;
    } else {
        --driver_it;
    }
    
    required_driver_version = driver_it->second;
    
    std::cout << "[RMagine - OptixContext] Init Optix (" << optix_version_str.str() << "). Required GPU driver >= " << driver_it->second / 100 << "." << driver_it->second % 100;
    
    if(driver_it->first != OPTIX_VERSION)
    {
        std::cout << " for Optix " << driver_it->first / 10000 << "." << (driver_it->first % 10000) / 100 << "." << driver_it->first % 100;
    }

    std::cout << std::endl;
    RM_OPTIX_CHECK( optixInit() );
    optix_initialized_ = true;
}

OptixContext::OptixContext(CudaContextPtr cuda_context)
:m_cuda_context(cuda_context)
{
    init(cuda_context);
    // std::cout << "[OptixContext::OptixContext()] constructed." << std::endl;
}

OptixContext::~OptixContext()
{
    optixDeviceContextDestroy( m_optix_context );
    // std::cout << "[OptixContext::~OptixContext()] destroyed." << std::endl;
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
    if(!optix_initialized())
    {
        optix_initialize();
    }

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    // 0: disable Setting the callback level will disable all messages. The callback function will not be called in this case
    // 1: fatal A non-recoverable error. The context and/or OptiX itself might no longer be in a usable state.
    // 2: error A recoverable error, e.g., when passing invalid call parameters. 
    // 3: warning Hints that OptiX might not behave exactly as requested by the user or may perform slower than expected. 
    // 4: print Status or progress messages.

    RM_OPTIX_CHECK( optixDeviceContextCreate( cuda_context->ref(), &options, &m_optix_context ) );
}

OptixContextPtr optix_def_ctx(new OptixContext(cuda_current_context()) );

OptixContextPtr optix_default_context()
{   
    return optix_def_ctx;
}

} // namespace rmagine