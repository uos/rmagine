#ifndef RMAGINE_UTIL_OPTIX_MODULES_H
#define RMAGINE_UTIL_OPTIX_MODULES_H

#include "optix_modules_def.h"
#include <vector>
#include <cuda.h>

namespace rmagine
{

// BASE CLASSES: SHADER INTERFACES

struct ProgramModule
{
    OptixModule_t*               module                 = nullptr;
    OptixModuleCompileOptions*   compile_options        = nullptr;

    ProgramModule();
    virtual ~ProgramModule();
};

struct ProgramGroup
{
    ProgramModulePtr            module          = nullptr;
    OptixProgramGroup_t*        prog_group      = nullptr;
    OptixProgramGroupOptions*   options         = nullptr;
    CUdeviceptr 	            record          = 0;
    unsigned int                record_stride   = 0;
    unsigned int                record_count    = 0;

    ProgramGroup();
    virtual ~ProgramGroup();
};

struct Pipeline
{
    OptixPipeline_t*    pipeline = nullptr;
    OptixShaderBindingTable* sbt = nullptr;
    OptixPipelineCompileOptions* compile_options = nullptr;

    Pipeline();

    virtual ~Pipeline();
};

} // namespace rmagine

#endif // RMAGINE_UTIL_OPTIX_MODULES_H