#include "rmagine/util/optix/optix_modules.h"


#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <rmagine/map/optix/OptixScene.hpp>


namespace rmagine {


ProgramModule::~ProgramModule()
{
    #if OPTIX_VERSION >= 70400
    if(compile_options.payloadTypes)
    {
        cudaFreeHost(compile_options.payloadTypes);
    }
    #endif

    if(module)
    {
        optixModuleDestroy( module );
    }
}

ProgramGroup::~ProgramGroup()
{
    if(record)
    {
        cudaFree( reinterpret_cast<void*>( record ) );
    }

    if(prog_group)
    {
        optixProgramGroupDestroy( prog_group );
    }
}


Pipeline::~Pipeline()
{
    if(pipeline)
    {
        optixPipelineDestroy( pipeline );
    }
}

// IMPLEMENTATIONS - TODO Move to simulation


} // namespace rmagine