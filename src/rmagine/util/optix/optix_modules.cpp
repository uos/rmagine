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
    if(compile_options->payloadTypes)
    {
        cudaFreeHost(compile_options->payloadTypes);
    }
    #endif

    if(module)
    {
        optixModuleDestroy( module );
    }

    delete compile_options;
}

void ProgramModule::compile(
    const OptixPipelineCompileOptions* pipeline_compile_options,
    OptixContextPtr ctx)
{
    if(ptx.empty())
    {
        throw std::runtime_error("ProgramModule - PTX empty");
    }

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    ctx->ref(),
                    compile_options,
                    pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &module
                    ));
}


ProgramGroup::ProgramGroup()
:options(new OptixProgramGroupOptions({}))
,description(new OptixProgramGroupDesc({}))
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

    delete description;
    delete options;
}

void ProgramGroup::create(OptixContextPtr ctx)
{
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        ctx->ref(),
                        description,
                        1,   // num program groups
                        options,
                        log,
                        &sizeof_log,
                        &prog_group
                        ) );
}

Pipeline::Pipeline()
:sbt(new OptixShaderBindingTable({}))
,compile_options(new OptixPipelineCompileOptions({}))
,link_options(new OptixPipelineLinkOptions({}))
{
    
}

Pipeline::~Pipeline()
{
    if(pipeline)
    {
        optixPipelineDestroy( pipeline );
    }

    delete link_options;
    delete compile_options;
    delete sbt;
}

void Pipeline::create(
    OptixContextPtr ctx)
{
    std::vector<OptixProgramGroup> program_groups_;

    for(auto elem : prog_groups)
    {
        program_groups_.push_back(elem->prog_group);
    }

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
        ctx->ref(),
        compile_options,
        link_options,
        &program_groups_[0],
        program_groups_.size(),
        log,
        &sizeof_log,
        &pipeline
        ) );
}


} // namespace rmagine