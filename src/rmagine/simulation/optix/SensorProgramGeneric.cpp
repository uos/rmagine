#include "rmagine/simulation/optix/SensorProgramGeneric.hpp"

#include "rmagine/util/GenericAlign.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"
#include "rmagine/simulation/optix/OptixSimulationData.hpp"

#include "rmagine/map/optix/OptixInstances.hpp"
#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/map/optix/OptixInst.hpp"

// use own lib instead
#include "rmagine/util/optix/OptixUtil.hpp"
#include "rmagine/util/optix/OptixSbtRecord.hpp"
#include "rmagine/util/optix/OptixData.hpp"

#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <iostream>
#include <fstream>

#include <rmagine/util/StopWatch.hpp>


namespace rmagine {

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

static std::string raygen_ptx_from_model_type(unsigned int model_type)
{
    std::string ptx;

    if(model_type == 0)
    {
        const char *kernel =
        #include "kernels/SphereProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 1) {
        const char *kernel =
        #include "kernels/PinholeProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 2) {
        const char *kernel =
        #include "kernels/O1DnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 3) {
        const char *kernel =
        #include "kernels/OnDnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else {
        std::cout << "[SensorProgramGeneric::raygen_ptx_from_model_type] ERROR model_type " << model_type << " not supported!" << std::endl;
        throw std::runtime_error("[SensorProgramGeneric::raygen_ptx_from_model_type] ERROR loading ptx");
    }

    return ptx;
}

SensorProgramGeneric::SensorProgramGeneric(
    OptixMapPtr map,
    const OptixSimulationDataGeneric& flags)
:SensorProgramGeneric(map->scene(), flags)
{
    
}

SensorProgramGeneric::SensorProgramGeneric(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags)
{
    OptixSceneType scene_type = scene->type();
    unsigned int scene_depth = scene->depth();

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );


    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;


    pipeline_compile_options.traversableGraphFlags = scene->traversableGraphFlags();
    
    // max payload values: 32
    pipeline_compile_options.numPayloadValues      = 0;
    // if dont use module payloads:
    // pipeline_compile_options.numPayloadValues      = 8;
    pipeline_compile_options.numAttributeValues    = 2;
#ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    unsigned int semantics[8] = {
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };

    // max payload values: 8
    OptixPayloadType payloadType;
    payloadType.numPayloadValues = 8;
    payloadType.payloadSemantics = semantics;


    { // GEN MODULE
        std::string ptx = raygen_ptx_from_model_type(flags.model_type);

        if(ptx.empty())
        {
            throw std::runtime_error("ScanProgramRanges could not find its PTX part");
        }

        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

        module_compile_options.numPayloadTypes = 1;
        module_compile_options.payloadTypes = &payloadType;

        OPTIX_CHECK( optixModuleCreateFromPTX(
                scene->context()->ref(),
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module_gen
                ));
    }

    { // HIT MODULE 
        const char* kernel = 
        #include "kernels/ProgramHitString.h"
        ;

        std::vector<OptixModuleCompileBoundValueEntry> options = make_bounds(flags);

        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    #ifndef NDEBUG
        // std::cout << "OPTIX_COMPILE_DEBUG_LEVEL_FULL" << std::endl;
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #else
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #endif
        module_compile_options.boundValues = &options[0];
        module_compile_options.numBoundValues = options.size();

        module_compile_options.numPayloadTypes = 1;
        module_compile_options.payloadTypes = &payloadType;

        std::string ptx(kernel);

        OPTIX_CHECK( optixModuleCreateFromPTX(
                    scene->context()->ref(),
                    &module_compile_options,
                    &pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &module_hit
                    ));
    }

    // 2. initProgramGroups
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros
    program_group_options.payloadType = &payloadType;

    // 2.1. RAYGEN
    {
        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module_gen;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    scene->context()->ref(),
                    &raygen_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group
                    ) );
    }

    // 2.2 Miss programs
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};

        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module_hit;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                scene->context()->ref(),
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
                ));
    }
    
    // 2.3 Closest Hit programs
    {

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};

        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module_hit;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                scene->context()->ref(),
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
                ));
    }


    // std::cout << "SCENE DEPTH: " << scene_depth << std::endl;

    // 3. link pipeline
    // traverse depth = 2 for ias + gas
    uint32_t    max_traversable_depth = scene_depth;
    const uint32_t    max_trace_depth  = 1; // TODO: 31 is maximum. Set this dynamically?
    
    OptixProgramGroup program_groups[] = { 
        raygen_prog_group, 
        miss_prog_group, 
        hitgroup_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth;
#ifndef NDEBUG
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
#endif
    OPTIX_CHECK_LOG( optixPipelineCreate(
                scene->context()->ref(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                log,
                &sizeof_log,
                &pipeline
                ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                0,  // maxCCDepth
                                                0,  // maxDCDEpth
                                                &direct_callable_stack_size_from_traversal,
                                                &direct_callable_stack_size_from_state, 
                                                &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            max_traversable_depth  // maxTraversableDepth
                                            ) );

    // std::cout << "Construct SBT ..." << std::endl;
    // 4. setup shader binding table

    m_scene = scene;

    // must be received from scene
    const size_t n_miss_record = 1;
    const size_t n_hitgroup_records = scene->requiredSBTEntries();
    

    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = n_miss_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = n_hitgroup_records;
    
    const size_t raygen_record_size     = sizeof( RayGenSbtRecord );
    const size_t miss_record_size       = sbt.missRecordStrideInBytes * sbt.missRecordCount;
    const size_t hitgroup_record_size   = sbt.hitgroupRecordStrideInBytes * sbt.hitgroupRecordCount;

    // malloc
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.raygenRecord ), raygen_record_size ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.missRecordBase ), miss_record_size) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.hitgroupRecordBase ), hitgroup_record_size ) );

    if(rg_sbt.size() < 1)
    {
        rg_sbt.resize(1);
        OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt[0] ) );
    }

    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( sbt.raygenRecord ),
                rg_sbt.raw(),
                raygen_record_size,
                cudaMemcpyHostToDevice,
                m_scene->stream()->handle()
                ) );
    
    updateSBT();
}

SensorProgramGeneric::~SensorProgramGeneric()
{
    // std::cout << "Destruct SensorProgramGeneric" << std::endl;
    optixModuleDestroy( module_gen );
    optixModuleDestroy( module_hit );
}

void SensorProgramGeneric::updateSBT()
{
    const size_t n_hitgroups_required = m_scene->requiredSBTEntries(); 

    if(n_hitgroups_required > sbt.hitgroupRecordCount)
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.hitgroupRecordBase ), n_hitgroups_required * sbt.hitgroupRecordStrideInBytes ) );
        sbt.hitgroupRecordCount = n_hitgroups_required;
    }

    const size_t raygen_record_size     = sizeof( RayGenSbtRecord );
    const size_t miss_record_size       = sbt.missRecordStrideInBytes * sbt.missRecordCount;
    const size_t hitgroup_record_size   = sbt.hitgroupRecordStrideInBytes * sbt.hitgroupRecordCount;
    

    

    if(ms_sbt.size() < sbt.missRecordCount)
    {
        ms_sbt.resize(sbt.missRecordCount);
        for(size_t i=0; i<sbt.missRecordCount; i++)
        {
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt[i] ) );
        }
    }
    
    if(hg_sbt.size() < sbt.hitgroupRecordCount)
    {
        hg_sbt.resize(sbt.hitgroupRecordCount);
        for(size_t i=0; i<sbt.hitgroupRecordCount; i++)
        {
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt[i] ) );
        }
    }

    for(size_t i=0; i<hg_sbt.size(); i++)
    {
        hg_sbt[i].data = m_scene->sbt_data;
    }

    // upload
    

    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( sbt.missRecordBase ),
                ms_sbt.raw(),
                miss_record_size,
                cudaMemcpyHostToDevice,
                m_scene->stream()->handle()
                ) );
    
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                hg_sbt.raw(),
                hitgroup_record_size,
                cudaMemcpyHostToDevice,
                m_scene->stream()->handle()
                ) );
}

} // namespace rmagine