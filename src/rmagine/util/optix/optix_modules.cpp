#include "rmagine/util/optix/optix_modules.h"


#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <rmagine/map/optix/OptixScene.hpp>

#include <optix.h>


namespace rmagine {


ProgramModule::ProgramModule()
:compile_options(new OptixModuleCompileOptions({}))
{

}

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

    delete compile_options;
}


ProgramGroup::ProgramGroup()
:options(new OptixProgramGroupOptions({}))
{

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

    delete options;
}


Pipeline::Pipeline()
:sbt(new OptixShaderBindingTable({}))
,compile_options(new OptixPipelineCompileOptions({}))
{
    
}

Pipeline::~Pipeline()
{
    if(pipeline)
    {
        optixPipelineDestroy( pipeline );
    }

    delete compile_options;
    delete sbt;
}

// IMPLEMENTATIONS - TODO Move to simulation


} // namespace rmagine