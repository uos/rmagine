#include "rmagine/simulation/optix/sim_pipelines.h"
#include "rmagine/simulation/optix/sim_program_groups.h"

#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/simulation/optix/OptixProgramMap.hpp>

namespace rmagine
{

SimPipeline::~SimPipeline()
{
    // std::cout << "[SimPipeline::~SimPipeline()] destroyed." << std::endl;
}

void SimPipeline::onDepthChanged()
{
    // std::cout << "[SimPipeline::onDepthChanged]" << std::endl;
    // TODO
}

void SimPipeline::onCommitDone(
    const OptixSceneCommitResult& info)
{
    // std::cout << "[SimPipeline::onCommitDone]" << std::endl;
    // TODO
}

// cache

std::unordered_map<OptixSceneWPtr,
    std::unordered_map<OptixSimulationDataGeneric, SimPipelinePtr>
> pipeline_sim_cache;

SimPipelinePtr make_pipeline_sim(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags)
{
    auto scene_it = pipeline_sim_cache.find(scene);

    if(scene_it != pipeline_sim_cache.end())
    {
        auto flags_it = scene_it->second.find(flags);
        if(flags_it != scene_it->second.end())
        {
            return flags_it->second;
        }
    } else {
        pipeline_sim_cache[scene] = {};
    }

    unsigned int traversable_graph_flags = scene->traversableGraphFlags();

    SimPipelinePtr ret = std::make_shared<SimPipeline>();

    ret->raygen = make_program_group_sim_gen(scene, flags.model_type);
    ret->miss = make_program_group_sim_miss(scene, flags);
    ret->hit = make_program_group_sim_hit(scene, flags);

    ret->sbt->raygenRecord                = ret->raygen->record;

    ret->sbt->missRecordBase              = ret->miss->record;
    ret->sbt->missRecordStrideInBytes     = ret->miss->record_stride;
    ret->sbt->missRecordCount             = ret->miss->record_count;

    ret->sbt->hitgroupRecordBase          = ret->hit->record;
    ret->sbt->hitgroupRecordStrideInBytes = ret->hit->record_stride;
    ret->sbt->hitgroupRecordCount         = ret->hit->record_count;

    // scene->

    uint32_t          max_traversable_depth = scene->depth();
    const uint32_t    max_trace_depth  = 1; // TODO: 31 is maximum. Set this dynamically?

    OptixProgramGroup program_groups[] = { 
        ret->raygen->prog_group, 
        ret->miss->prog_group, 
        ret->hit->prog_group
    };


    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth;
    #ifndef NDEBUG
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #else
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
    #endif

    {
        ret->compile_options->usesMotionBlur        = false;

        ret->compile_options->traversableGraphFlags = traversable_graph_flags;
        
        // max payload values: 32
        #if OPTIX_VERSION >= 70400
        ret->compile_options->numPayloadValues      = 0;
        #else
        // if dont use module payloads: cannot use module paypload for Optix < 7.4
        ret->compile_options->numPayloadValues      = 8;
        #endif
        

        // pipeline_compile_options.numPayloadValues      = 8;
        ret->compile_options->numAttributeValues    = 2;
    #ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        ret->compile_options->exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    #else
        ret->compile_options->exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    #endif
        ret->compile_options->pipelineLaunchParamsVariableName = "mem";
        ret->compile_options->usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
        scene->context()->ref(),
        ret->compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &ret->pipeline
        ) );

    scene->addEventReceiver(ret);

    // caching is important. we dont want to receive events from scene more than once per commit
    pipeline_sim_cache[scene][flags] = ret;

    return ret;
}

} // namespace rmagine