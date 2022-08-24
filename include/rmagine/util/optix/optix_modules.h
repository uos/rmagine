#ifndef RMAGINE_UTIL_OPTIX_MODULES_H
#define RMAGINE_UTIL_OPTIX_MODULES_H

#include <optix.h>
#include <memory>

#include <rmagine/util/hashing.h>

#include <vector>


namespace rmagine
{

// BASE CLASSES: SHADER INTERFACES

struct ProgramModule
{
    OptixModule                 module                 = nullptr;
    OptixModuleCompileOptions   compile_options        = {};

    virtual ~ProgramModule();
};

using ProgramModulePtr = std::shared_ptr<ProgramModule>;

using ProgramModuleWPtr = std::weak_ptr<ProgramModule>;


struct ProgramGroup
{
    ProgramModulePtr            module          = nullptr;
    OptixProgramGroup           prog_group      = nullptr;
    OptixProgramGroupOptions    options = {};
    CUdeviceptr 	            record          = 0;
    unsigned int                record_stride   = 0;
    unsigned int                record_count    = 0;

    virtual ~ProgramGroup();
};

using ProgramGroupPtr = std::shared_ptr<ProgramGroup>;

struct Pipeline
{
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};

    OptixPipelineCompileOptions pipeline_compile_options = {};

    ~Pipeline();
};

using PipelinePtr = std::shared_ptr<Pipeline>;

} // namespace rmagine

namespace std
{

// INSTANCE
template<>
struct hash<rmagine::ProgramModuleWPtr> 
    : public rmagine::weak_hash<rmagine::ProgramModule>
{};

template<>
struct equal_to<rmagine::ProgramModuleWPtr> 
    : public rmagine::weak_equal_to<rmagine::ProgramModule>
{};

template<>
struct less<rmagine::ProgramModuleWPtr> 
    : public rmagine::weak_less<rmagine::ProgramModule>
{};

} // namespace std

#endif // RMAGINE_UTIL_OPTIX_MODULES_H