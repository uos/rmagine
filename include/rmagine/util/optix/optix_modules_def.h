#ifndef RMAGINE_UTIL_OPTIX_MODULES_DEF_H
#define RMAGINE_UTIL_OPTIX_MODULES_DEF_H

#include <memory>
#include <rmagine/util/hashing.h>


struct OptixModule_t;
struct OptixProgramGroup_t;
struct OptixPipeline_t;
struct OptixDenoiser_t;
struct OptixShaderBindingTable;

// Options
struct OptixModuleCompileOptions;
struct OptixProgramGroupOptions;
struct OptixPipelineCompileOptions;



namespace rmagine
{

struct ProgramModule;
struct ProgramGroup;
struct Pipeline;

using ProgramModulePtr = std::shared_ptr<ProgramModule>;
using ProgramGroupPtr = std::shared_ptr<ProgramGroup>;
using PipelinePtr = std::shared_ptr<Pipeline>;

using ProgramModuleWPtr = std::weak_ptr<ProgramModule>;

} // namespace rmagine

namespace std
{

// ProgramModule
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


#endif // RMAGINE_UTIL_OPTIX_MODULES_DEF_H