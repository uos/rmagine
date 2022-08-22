#include "rmagine/simulation/optix/O1DnProgramGeneric.hpp"

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

namespace rmagine {

O1DnProgramGeneric::O1DnProgramGeneric(
    OptixMapPtr map,
    const OptixSimulationDataGenericO1Dn& flags)
:O1DnProgramGeneric(map->scene(), flags)
{

}

O1DnProgramGeneric::O1DnProgramGeneric(
    OptixScenePtr scene,
    const OptixSimulationDataGenericO1Dn& flags)
{
    const char *kernel =
    #include "kernels/O1DnProgramGenericString.h"
    ;

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    
    OptixModuleCompileBoundValueEntry options[7];
    // computeHits
    options[0] = {};
    options[0].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeHits);
    options[0].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeHits );
    options[0].boundValuePtr = &flags.computeHits;
    // computeRanges
    options[1] = {};
    options[1].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeRanges);
    options[1].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeRanges );
    options[1].boundValuePtr = &flags.computeRanges;
    // computePoints
    options[2] = {};
    options[2].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computePoints);
    options[2].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computePoints );
    options[2].boundValuePtr = &flags.computePoints;
    // computeNormals
    options[3] = {};
    options[3].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeNormals);
    options[3].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeNormals );
    options[3].boundValuePtr = &flags.computeNormals;
    // computeFaceIds
    options[4] = {};
    options[4].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeFaceIds);
    options[4].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeFaceIds );
    options[4].boundValuePtr = &flags.computeFaceIds;
    // computeGeomIds
    options[5] = {};
    options[5].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeGeomIds);
    options[5].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeGeomIds );
    options[5].boundValuePtr = &flags.computeGeomIds;
    // computeObjectIds
    options[6] = {};
    options[6].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericO1Dn, computeObjectIds);
    options[6].sizeInBytes = sizeof( OptixSimulationDataGenericO1Dn::computeObjectIds );
    options[6].boundValuePtr = &flags.computeObjectIds;


    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    module_compile_options.boundValues = &options[0];
    module_compile_options.numBoundValues = 7;
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;

    
    OptixSceneType scene_type = scene->type();
    unsigned int scene_depth = scene->depth();

    if(scene_depth < 1)
    {
        std::cout << "ERROR: OptixScene is empty" << std::endl; 
        throw std::runtime_error("OptixScene has not root");
    } else if(scene_depth < 2) {
        // 1 Only GAS
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    } else if(scene_depth < 3) {
        // 2 Only single level IAS
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    } else {
        // 3 or more allow any
        // careful: with two level IAS performance is half as slow as single level IAS
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    }
    
    pipeline_compile_options.numPayloadValues      = 8;
    pipeline_compile_options.numAttributeValues    = 2;
#ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string ptx(kernel);

    if(ptx.empty())
    {
        throw std::runtime_error("ScanProgramRanges could not find its PTX part");
    }


    OPTIX_CHECK( optixModuleCreateFromPTX(
                scene->context()->ref(),
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ));

    // std::cout << "Compiled Generic Shader" << std::endl;

    // 2. initProgramGroups
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros
    sizeof_log = sizeof( log );

    // 2.1. RAYGEN
    {  
        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
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
        miss_prog_group_desc.miss.module            = module;
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
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
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
    sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                scene->context()->ref(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
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
    const size_t n_hitgroup_records = scene->required_sbt_entries;
    

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

    updateSBT();
}


O1DnProgramGeneric::~O1DnProgramGeneric()
{
    // std::cout << "Destruct SphereProgramGeneric" << std::endl;
}

void O1DnProgramGeneric::updateSBT()
{
    const size_t n_hitgroups_required = m_scene->required_sbt_entries;   

    if(n_hitgroups_required > sbt.hitgroupRecordCount)
    {
        // std::cout << "RESIZE SBT to " << n_hitgroups_required << std::endl;
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.hitgroupRecordBase ), n_hitgroups_required * sbt.hitgroupRecordStrideInBytes ) );
        sbt.hitgroupRecordCount = n_hitgroups_required;
    }

    const size_t raygen_record_size     = sizeof( RayGenSbtRecord );
    const size_t miss_record_size       = sbt.missRecordStrideInBytes * sbt.missRecordCount;
    const size_t hitgroup_record_size   = sbt.hitgroupRecordStrideInBytes * sbt.hitgroupRecordCount;
    
    rg_sbt.resize(1);
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt[0] ) );

    ms_sbt.resize(sbt.missRecordCount);
    for(size_t i=0; i<sbt.missRecordCount; i++)
    {
        OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt[i] ) );
    }
    
    if(hg_sbt.size() < sbt.hitgroupRecordCount)
    {
        hg_sbt.resize(sbt.hitgroupRecordCount);
        for(size_t i=0; i<sbt.hitgroupRecordCount; i++)
        {
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt[i] ) );
            hg_sbt[i].data = m_scene->sbt_data;
        }
    } else {
        for(size_t i=0; i<sbt.hitgroupRecordCount; i++)
        {
            hg_sbt[i].data = m_scene->sbt_data;
        }
    }

    // upload
    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( sbt.raygenRecord ),
                rg_sbt.raw(),
                raygen_record_size,
                cudaMemcpyHostToDevice,
                m_scene->stream()->handle()
                )  );

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