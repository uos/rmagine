#ifndef RMAGINE_UTIL_OPTIX_MODULES_H
#define RMAGINE_UTIL_OPTIX_MODULES_H

#include "optix_modules_def.h"
#include <vector>
#include <cuda.h>

#include "OptixContext.hpp"

namespace rmagine
{

// BASE CLASSES: SHADER INTERFACES

struct ProgramModule
{
    OptixModule_t*               module                 = nullptr;
    OptixModuleCompileOptions*   compile_options        = nullptr;
    std::string                  ptx;

    ProgramModule();
    virtual ~ProgramModule();

    void compile(
        const OptixPipelineCompileOptions* pipeline_compile_options,
        OptixContextPtr ctx = optix_default_context());
};

struct ProgramGroup
{
    ProgramModulePtr            module          = nullptr;
    OptixProgramGroup_t*        prog_group      = nullptr;
    OptixProgramGroupOptions*   options         = nullptr;
    OptixProgramGroupDesc*      description     = nullptr;
    CUdeviceptr 	            record          = 0;
    unsigned int                record_stride   = 0;
    unsigned int                record_count    = 0;

    ProgramGroup();
    virtual ~ProgramGroup();

    void create(OptixContextPtr ctx = optix_default_context());
};

struct Pipeline
{
    OptixPipeline_t*                pipeline            = nullptr;
    OptixShaderBindingTable*        sbt                 = nullptr;
    OptixPipelineCompileOptions*    compile_options     = nullptr;
    OptixPipelineLinkOptions*       link_options        = nullptr;
    std::vector<ProgramGroupPtr>    prog_groups;

    Pipeline();
    virtual ~Pipeline();

    void create(
        OptixContextPtr ctx = optix_default_context());
};

} // namespace rmagine

#endif // RMAGINE_UTIL_OPTIX_MODULES_H