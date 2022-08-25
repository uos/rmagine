#include "rmagine/simulation/optix/sim_modules.h"

#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <rmagine/map/optix/OptixScene.hpp>

#include <rmagine/simulation/optix/OptixProgramMap.hpp>


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

SimRayGenProgramGroup::~SimRayGenProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
    // std::cout << "[SimRayGenProgramGroup::~SimRayGenProgramGroup()] destroyed." << std::endl;
}

SimMissProgramGroup::~SimMissProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
    // std::cout << "[SimMissProgramGroup::~SimMissProgramGroup()] destroyed." << std::endl;
}

SimHitProgramGroup::~SimHitProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
    // std::cout << "[SimHitProgramGroup::~SimHitProgramGroup()] destroyed." << std::endl;
}

void SimHitProgramGroup::onSBTUpdated(
    bool size_changed)
{
    OptixScenePtr scene = m_scene.lock();

    if(scene)
    {
        if(size_changed)
        {
            size_t n_hitgroups_required = scene->requiredSBTEntries();

            if(n_hitgroups_required > record_count)
            {
                if(record_h)
                {
                    CUDA_CHECK( cudaFreeHost( record_h ) );
                }
                
                CUDA_CHECK( cudaMallocHost( &record_h, n_hitgroups_required * record_stride ) );

                for(size_t i=0; i<n_hitgroups_required; i++)
                {
                    OPTIX_CHECK( optixSbtRecordPackHeader( prog_group, &record_h[i] ) );
                }

                if( record )
                {
                    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( record ) ) );
                }
                
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &record ), n_hitgroups_required * record_stride ) );

                record_count = n_hitgroups_required;
            }
        }

        for(size_t i=0; i<record_count; i++)
        {
            record_h[i].data = scene->sbt_data;
        }

        CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( record ),
                    record_h,
                    record_count * record_stride,
                    cudaMemcpyHostToDevice,
                    scene->stream()->handle()
                    ) );
    }
    
}


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

    std::string ptx;

    if(sensor_id == 0)
    {
        static const char* kernel =
        #include "kernels/SphereProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(sensor_id == 1) {
        const char *kernel =
        #include "kernels/PinholeProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(sensor_id == 2) {
        const char *kernel =
        #include "kernels/O1DnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(sensor_id == 3) {
        const char *kernel =
        #include "kernels/OnDnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else {
        std::cout << "[OptixScene::raygen_ptx_from_model_type] ERROR model_type " << sensor_id << " not supported!" << std::endl;
        throw std::runtime_error("[OptixScene::raygen_ptx_from_model_type] ERROR loading ptx");
    }

    if(ptx.empty())
    {
        throw std::runtime_error("OptixScene could not find its PTX part");
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

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    scene->context()->ref(),
                    ret->compile_options,
                    &pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &ret->module
                    ));

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

    std::string ptx(kernel);
    if(ptx.empty())
    {
        throw std::runtime_error("OptixScene could not find its PTX part");
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

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                    scene->context()->ref(),
                    ret->compile_options,
                    &pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &ret->module
                    ));

    sim_module_hit_miss_cache[traversable_graph_flags][bid] = ret;

    return ret;
}

std::unordered_map<OptixSceneWPtr, 
    std::unordered_map<ProgramModuleWPtr, SimRayGenProgramGroupPtr>
> m_program_group_sim_gen_cache;

SimRayGenProgramGroupPtr make_program_group_sim_gen(
    OptixScenePtr scene,
    ProgramModulePtr module)
{
    auto scene_it = m_program_group_sim_gen_cache.find(scene);

    if(scene_it != m_program_group_sim_gen_cache.end())
    {
        auto module_it = scene_it->second.find(module);
        if(module_it != scene_it->second.end())
        {
            return module_it->second;
        }
    } else {
        m_program_group_sim_gen_cache[scene] = {};
    }

    SimRayGenProgramGroupPtr ret = std::make_shared<SimRayGenProgramGroup>();

    OptixProgramGroupDesc prog_group_desc    = {}; //
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc.raygen.module            = module->module;
    prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif
    ret->module = module;

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        scene->context()->ref(),
                        &prog_group_desc,
                        1,   // num program groups
                        ret->options,
                        log,
                        &sizeof_log,
                        &ret->prog_group
                        ) );

    { // init SBT Records
        const size_t raygen_record_size     = sizeof( SimRayGenProgramGroup::SbtRecordData );
        
        CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            raygen_record_size ) );

        OPTIX_CHECK( optixSbtRecordPackHeader( 
            ret->prog_group,
            &ret->record_h[0] ) );

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), raygen_record_size ) );

        CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( ret->record ),
                    ret->record_h,
                    raygen_record_size,
                    cudaMemcpyHostToDevice,
                    scene->stream()->handle()
                    ) );

        ret->record_stride = sizeof( SimRayGenProgramGroup::SbtRecordData );
        ret->record_count = 1;
    }

    m_program_group_sim_gen_cache[scene][module] = ret;

    return ret;
}

SimRayGenProgramGroupPtr make_program_group_sim_gen(
    OptixScenePtr scene,
    unsigned int sensor_id)
{
    ProgramModulePtr module = make_program_module_sim_gen(scene, sensor_id);
    return make_program_group_sim_gen(scene, module);
}

std::unordered_map<OptixSceneWPtr, 
    std::unordered_map<ProgramModuleWPtr, SimMissProgramGroupPtr>
> m_program_group_sim_miss_cache;

SimMissProgramGroupPtr make_program_group_sim_miss(
    OptixScenePtr scene,
    ProgramModulePtr module)
{
    auto scene_it = m_program_group_sim_miss_cache.find(scene);

    if(scene_it != m_program_group_sim_miss_cache.end())
    {
        auto module_it = scene_it->second.find(module);
        if(module_it != scene_it->second.end())
        {
            return module_it->second;
        }
    } else {
        m_program_group_sim_miss_cache[scene] = {};
    }

    SimMissProgramGroupPtr ret = std::make_shared<SimMissProgramGroup>();

    OptixProgramGroupDesc prog_group_desc    = {}; //
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc.raygen.module            = module->module;
    prog_group_desc.raygen.entryFunctionName = "__miss__ms";

    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif
    ret->module = module;

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        scene->context()->ref(),
                        &prog_group_desc,
                        1,   // num program groups
                        ret->options,
                        log,
                        &sizeof_log,
                        &ret->prog_group
                        ) );

    { // init SBT Records
        const size_t miss_record_size     = sizeof( SimMissProgramGroup::SbtRecordData );
        
        CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            miss_record_size ) );

        OPTIX_CHECK( optixSbtRecordPackHeader( 
            ret->prog_group,
            &ret->record_h[0] ) );

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), miss_record_size ) );

        CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( ret->record ),
                    ret->record_h,
                    miss_record_size,
                    cudaMemcpyHostToDevice,
                    scene->stream()->handle()
                    ) );

        ret->record_stride = sizeof( SimMissProgramGroup::SbtRecordData );
        ret->record_count = 1;
    }

    m_program_group_sim_miss_cache[scene][module] = ret;

    return ret;
}

SimMissProgramGroupPtr make_program_group_sim_miss(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags)
{
    ProgramModulePtr module = make_program_module_sim_hit_miss(scene, flags);
    return make_program_group_sim_miss(scene, module);
}

std::unordered_map<OptixSceneWPtr, 
    std::unordered_map<ProgramModuleWPtr, SimHitProgramGroupPtr>
> m_program_group_sim_hit_cache;

SimHitProgramGroupPtr make_program_group_sim_hit(
    OptixScenePtr scene,
    ProgramModulePtr module)
{
    auto scene_it = m_program_group_sim_hit_cache.find(scene);

    if(scene_it != m_program_group_sim_hit_cache.end())
    {
        auto module_it = scene_it->second.find(module);
        if(module_it != scene_it->second.end())
        {
            return module_it->second;
        }
    } else {
        m_program_group_sim_hit_cache[scene] = {};
    }

    SimHitProgramGroupPtr ret = std::make_shared<SimHitProgramGroup>();

    OptixProgramGroupDesc prog_group_desc    = {}; //
    prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.raygen.module            = module->module;
    prog_group_desc.raygen.entryFunctionName = "__closesthit__ch";

    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif
    ret->module = module;

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        scene->context()->ref(),
                        &prog_group_desc,
                        1,   // num program groups
                        ret->options,
                        log,
                        &sizeof_log,
                        &ret->prog_group
                        ) );

    { // init SBT Records
        const size_t n_hitgroup_records = scene->requiredSBTEntries();   
        const size_t hitgroup_record_size     = sizeof( SimMissProgramGroup::SbtRecordData ) * n_hitgroup_records;
        
        CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            hitgroup_record_size ) );

        for(size_t i=0; i<n_hitgroup_records; i++)
        {
            OPTIX_CHECK( optixSbtRecordPackHeader( 
                ret->prog_group,
                &ret->record_h[i] ) );
            ret->record_h[i].data = scene->sbt_data;
        }
        
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), hitgroup_record_size ) );

        CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( ret->record ),
                    ret->record_h,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice,
                    scene->stream()->handle()
                    ) );

        ret->record_stride = sizeof( SimMissProgramGroup::SbtRecordData );
        ret->record_count = n_hitgroup_records;
    }

    scene->addEventReceiver(ret);

    m_program_group_sim_hit_cache[scene][module] = ret;

    return ret;
}

SimHitProgramGroupPtr make_program_group_sim_hit(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags)
{
    ProgramModulePtr module = make_program_module_sim_hit_miss(scene, flags);
    return make_program_group_sim_hit(scene, module);
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