#include "rmagine/simulation/optix/sim_modules.h"
#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/simulation/optix/common.h>


namespace rmagine
{

static std::vector<OptixModuleCompileBoundValueEntry> make_bounds(
    const OptixSimulationDataGeneric& flags)
{
    std::vector<OptixModuleCompileBoundValueEntry> options;
        
    { // computeHits
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeHits);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeHits );
        option.boundValuePtr = &flags.computeHits;
        options.push_back(option);
    }
    
    { // computeRanges
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeRanges);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeRanges );
        option.boundValuePtr = &flags.computeRanges;
        options.push_back(option);
    }

    { // computePoints
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computePoints);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computePoints );
        option.boundValuePtr = &flags.computePoints;
        options.push_back(option);
    }

    { // computeNormals
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeNormals);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeNormals );
        option.boundValuePtr = &flags.computeNormals;
        options.push_back(option);
    }

    { // computeFaceIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeFaceIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeFaceIds );
        option.boundValuePtr = &flags.computeFaceIds;
        options.push_back(option);
    }

    { // computeGeomIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeGeomIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeGeomIds );
        option.boundValuePtr = &flags.computeGeomIds;
        options.push_back(option);
    }

    { // computeObjectIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeObjectIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeObjectIds );
        option.boundValuePtr = &flags.computeObjectIds;
        options.push_back(option);
    }

    return options;
}

// cache per (traversableGraphFlags ,scene)
std::unordered_map<unsigned int, 
    std::unordered_map<unsigned int, ProgramModulePtr> 
> sim_module_gen_cache;

ProgramModulePtr make_program_module_sim_gen(
    OptixScenePtr scene, 
    unsigned int sensor_id)
{
    unsigned int traversable_graph_flags = scene->traversableGraphFlags();
    auto scene_it = sim_module_gen_cache.find(traversable_graph_flags);

    if(scene_it != sim_module_gen_cache.end())
    {
        // found scene in cache
        auto sensor_it = scene_it->second.find(sensor_id);
        if(sensor_it != scene_it->second.end())
        {
            return sensor_it->second;
        }
    } else {
        // scene not found in cache -> creating empty new. filled later
        sim_module_gen_cache[traversable_graph_flags] = {};
    }

    ProgramModulePtr ret = std::make_shared<ProgramModule>();

    ret->compile_options->maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    ret->compile_options->optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    ret->compile_options->optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    #if OPTIX_VERSION >= 70400
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #else
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    #endif
#endif


    #if OPTIX_VERSION >= 70400
    ret->compile_options->numPayloadTypes = 1;
    CUDA_CHECK(cudaMallocHost(&ret->compile_options->payloadTypes, sizeof(OptixPayloadType) ) );
    
    ret->compile_options->payloadTypes[0].numPayloadValues = 8;
    ret->compile_options->payloadTypes[0].payloadSemantics = (const unsigned int[8]) {
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };
    #endif
    
    ProgramModulePtr module = std::make_shared<ProgramModule>();

    if(sensor_id == 0)
    {
        static const char* kernel =
        #include "kernels/SphereProgramGenString.h"
        ;
        ret->ptx = std::string(kernel);
    } else if(sensor_id == 1) {
        const char *kernel =
        #include "kernels/PinholeProgramGenString.h"
        ;
        ret->ptx = std::string(kernel);
    } else if(sensor_id == 2) {
        const char *kernel =
        #include "kernels/O1DnProgramGenString.h"
        ;
        ret->ptx = std::string(kernel);
    } else if(sensor_id == 3) {
        const char *kernel =
        #include "kernels/OnDnProgramGenString.h"
        ;
        ret->ptx = std::string(kernel);
    } else {
        std::cout << "[OptixScene::raygen_ptx_from_model_type] ERROR model_type " << sensor_id << " not supported!" << std::endl;
        throw std::runtime_error("[OptixScene::raygen_ptx_from_model_type] ERROR loading ptx");
    }

    // TODO: share this between nearly any module?
    // depends on:
    // - scene : traversableGraphFlags
    // - numPayloadValues: modules/shader
    // - numAttributeValues: modules/shader
    // is used in:
    // - optixModuleCreateFromPTX
    // - optixPipelineCreate
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        pipeline_compile_options.usesMotionBlur        = false;


        pipeline_compile_options.traversableGraphFlags = traversable_graph_flags;
        
        // max payload values: 32
        #if OPTIX_VERSION >= 70400
            pipeline_compile_options.numPayloadValues      = 0;
        #else
            // if dont use module payloads: OptiX < 7.4 cannot use them
            pipeline_compile_options.numPayloadValues      = 8;
        #endif
        pipeline_compile_options.numAttributeValues    = 2;
    #ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    #else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    #endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }

    ret->compile(&pipeline_compile_options, scene->context());

    // cache
    sim_module_gen_cache[traversable_graph_flags][sensor_id] = ret; 

    return ret;
}

std::unordered_map<unsigned int, 
    std::unordered_map<unsigned int, ProgramModulePtr> 
> sim_module_hit_miss_cache;

ProgramModulePtr make_program_module_sim_hit_miss(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags)
{
    unsigned int traversable_graph_flags = scene->traversableGraphFlags();
    unsigned int bid = boundingId(flags);


    auto scene_it = sim_module_hit_miss_cache.find(traversable_graph_flags);
    if(scene_it != sim_module_hit_miss_cache.end())
    {
        auto bound_it = scene_it->second.find(bid);
        if(bound_it != scene_it->second.end())
        {
            return bound_it->second;
        }
    } else {
        sim_module_hit_miss_cache[traversable_graph_flags] = {};
    }

    ProgramModulePtr ret = std::make_shared<ProgramModule>();


    std::vector<OptixModuleCompileBoundValueEntry> bounds = make_bounds(flags);
    static const char* kernel = 
    #include "kernels/SensorProgramHitString.h"
    ;

    ret->compile_options->maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

#ifndef NDEBUG
    ret->compile_options->optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    ret->compile_options->optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    #if OPTIX_VERSION >= 70400
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #else
    ret->compile_options->debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    #endif
#endif
    ret->compile_options->boundValues = &bounds[0];
    ret->compile_options->numBoundValues = bounds.size();

    #if OPTIX_VERSION >= 70400
    ret->compile_options->numPayloadTypes = 1;
    CUDA_CHECK(cudaMallocHost(&ret->compile_options->payloadTypes, sizeof(OptixPayloadType) ) );
    
    ret->compile_options->payloadTypes[0]->numPayloadValues = 8;
    ret->compile_options->payloadTypes[0]->payloadSemantics = (const unsigned int[8]) {
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };
    #endif

    ret->ptx = std::string(kernel);

    // TODO: share this between nearly any module?
    // depends on:
    // - scene : traversableGraphFlags
    // - numPayloadValues: modules/shader
    // - numAttributeValues: modules/shader
    // is used in:
    // - optixModuleCreateFromPTX
    // - optixPipelineCreate
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = traversable_graph_flags;
        
        #if OPTIX_VERSION >= 70400
            // use module payloads: can specify semantics more accurately
            pipeline_compile_options.numPayloadValues      = 0;
        #else
            // if dont use module payloads: OptiX < 7.4 cannot use module payloads
            // max payload values: 32
            pipeline_compile_options.numPayloadValues       = 8;
        #endif 
        
        // pipeline_compile_options.numPayloadValues      = 8;
        pipeline_compile_options.numAttributeValues    = 2;
    #ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    #else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    #endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    ret->compile(&pipeline_compile_options, scene->context());

    sim_module_hit_miss_cache[traversable_graph_flags][bid] = ret;

    return ret;
}

} // namespace rmagine