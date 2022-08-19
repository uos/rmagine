#include "rmagine/simulation/optix/SphereProgramGeneric.hpp"

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

SphereProgramGeneric::SphereProgramGeneric(
    OptixMapPtr map,
    const OptixSimulationDataGenericSphere& flags)
{   
    const char *kernel =
    #include "kernels/SphereProgramGenericString.h"
    ;

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    
    OptixModuleCompileBoundValueEntry options[7];
    // computeHits
    options[0] = {};
    options[0].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeHits);
    options[0].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeHits );
    options[0].boundValuePtr = &flags.computeHits;
    // computeRanges
    options[1] = {};
    options[1].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeRanges);
    options[1].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeRanges );
    options[1].boundValuePtr = &flags.computeRanges;
    // computePoints
    options[2] = {};
    options[2].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computePoints);
    options[2].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computePoints );
    options[2].boundValuePtr = &flags.computePoints;
    // computeNormals
    options[3] = {};
    options[3].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeNormals);
    options[3].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeNormals );
    options[3].boundValuePtr = &flags.computeNormals;
    // computeFaceIds
    options[4] = {};
    options[4].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeFaceIds);
    options[4].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeFaceIds );
    options[4].boundValuePtr = &flags.computeFaceIds;
    // computeFaceIds
    options[5] = {};
    options[5].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeGeomIds);
    options[5].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeGeomIds );
    options[5].boundValuePtr = &flags.computeGeomIds;
    // computeObjectIds
    options[6] = {};
    options[6].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeObjectIds);
    options[6].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeObjectIds );
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


    OptixScenePtr scene = map->scene();
    OptixSceneType scene_type = scene->type();
    unsigned int scene_depth = scene->depth();

    if(scene_depth < 1)
    {
        std::cout << "ERROR: OptixScene has not root" << std::endl; 
        throw std::runtime_error("OptixScene has not root");
    } else if(scene_depth < 2) {
        // 1 Only GAS
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    } else if(scene_depth < 3) {
        // 2 Only single level IAS
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    } else {
        std::cout << "ALLOW ANY" << std::endl;
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
        
        // if(scene_depth >= 3)
        // {
        //     std::cout << "ALLOW ANYHIT!" << std::endl;
        //     hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        //     hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        // }
        

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
                                                &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            max_traversable_depth  // maxTraversableDepth
                                            ) );

    // std::cout << "Construct SBT ..." << std::endl;
    // 4. setup shader binding table

    // fill Headers
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );

    const size_t raygen_record_size     = sizeof( RayGenSbtRecord );
    const size_t miss_record_size       = sizeof( MissSbtRecord );
    const size_t hitgroup_record_size   = sizeof( HitGroupSbtRecord );


    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = 1;

    // malloc
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.raygenRecord ), raygen_record_size ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.missRecordBase ), miss_record_size) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.hitgroupRecordBase ), hitgroup_record_size ) );
    
    m_scene = scene;
    m_map = map;

    updateSBT();
}

SphereProgramGeneric::~SphereProgramGeneric()
{
    // std::cout << "Destruct SphereProgramGeneric" << std::endl;
}

void SphereProgramGeneric::updateSBT()
{
    const size_t      raygen_record_size = sizeof( RayGenSbtRecord );
    const size_t      miss_record_size = sizeof( MissSbtRecord );
    const size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    
    if(m_scene->m_h_hitgroup_data.size() == 0)
    {
        std::cout << "[SphereProgramGeneric] ERROR no sbt data in scene. Did you call commit() on the scene first?" << std::endl;
    }

    hg_sbt.data = m_scene->m_h_hitgroup_data[0];

    // upload
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.raygenRecord ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.missRecordBase ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );
    
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );
}


} // namespace rmagine