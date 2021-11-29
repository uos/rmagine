#include "imagine/simulation/optix/ScanProgramGeneric.hpp"

#include "imagine/util/GenericAlign.hpp"
#include "imagine/util/optix/OptixDebug.hpp"
#include "imagine/simulation/optix/OptixSimulationData.hpp"

// use own lib instead
#include "imagine/util/optix/OptixUtil.hpp"
#include "imagine/util/optix/OptixSbtRecord.hpp"
#include "imagine/util/optix/OptixData.hpp"

#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <iostream>
#include <fstream>

namespace imagine {

typedef SbtRecord<RayGenDataEmpty>     RayGenSbtRecord;
typedef SbtRecord<MissDataEmpty>       MissSbtRecord;
typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;

ScanProgramGeneric::ScanProgramGeneric(OptixMapPtr map)
{
    const char *kernel =
    #include "kernels/ScanProgramGenericString.h"
    ;

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    
    // Try to enable something during compilation
    // OptixSimulationDataGeneric tmp_mem;
    // tmp_mem.computeHits   = true;
    // tmp_mem.computeRanges = true;
    // tmp_mem.computePoints = true;
    // tmp_mem.computeNormals = true;
    // tmp_mem.computeFaceIds = true;
    // tmp_mem.computeObjectIds = true;

    // determine number of payload values
    // int numPayloadValues = 8;
    // if(tmp_mem.computeRanges)
    // {
    //     numPayloadValues = std::max(numPayloadValues, 1);
    // }
    // if(tmp_mem.computePoints)
    // {
    //     numPayloadValues = std::max(numPayloadValues, 1);
    // }

//     OptixModuleCompileBoundValueEntry options[6];
//     // computeHits
//     options[0].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeHits);
//     options[0].sizeInBytes = sizeof( OptixSimulationDataGeneric::computeHits );
//     options[0].boundValuePtr = &tmp_mem.computeHits;
//     // computeRanges
//     options[1].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeRanges);
//     options[1].sizeInBytes = sizeof( OptixSimulationDataGeneric::computeRanges );
//     options[1].boundValuePtr = &tmp_mem.computeRanges;
//     // computePoints
//     options[2].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computePoints);
//     options[2].sizeInBytes = sizeof( OptixSimulationDataGeneric::computePoints );
//     options[2].boundValuePtr = &tmp_mem.computePoints;
//     // computeNormals
//     options[3].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeNormals);
//     options[3].sizeInBytes = sizeof( OptixSimulationDataGeneric::computeNormals );
//     options[3].boundValuePtr = &tmp_mem.computeNormals;
//     // computeFaceIds
//     options[4].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeFaceIds);
//     options[4].sizeInBytes = sizeof( OptixSimulationDataGeneric::computeFaceIds );
//     options[4].boundValuePtr = &tmp_mem.computeFaceIds;
//     // computeObjectIds
//     options[5].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeObjectIds);
//     options[5].sizeInBytes = sizeof( OptixSimulationDataGeneric::computeObjectIds );
//     options[5].boundValuePtr = &tmp_mem.computeObjectIds;


    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    // module_compile_options.boundValues = &options[0];
    // module_compile_options.numBoundValues = 6;
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;
    if(map->ias())
    {
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    } else {
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
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

    std::cout << "optixModuleCreateFromPTX" << std::endl;

    OPTIX_CHECK( optixModuleCreateFromPTX(
                map->context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ));

    std::cout << "initProgramGroups" << std::endl;

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
                    map->context,
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
        OptixProgramGroupDesc miss_prog_group_desc;

        miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ranges";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                map->context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_groups[0]
                ));

        miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__normals";   

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                map->context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_groups[1]
                ));     
    }
    
    // 2.3 Closest Hit programs
    {
        OptixProgramGroupDesc hitgroup_prog_group_desc;

        hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ranges";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                map->context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_groups[0]
                ));

        hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__normals";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                map->context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_groups[1]
                ));
    }

    std::cout << "link pipeline" << std::endl;
    // 3. link pipeline
    // traverse depth = 2 for ias + gas
    uint32_t    max_traversable_depth = 1;
    if(map->ias())
    {
        max_traversable_depth = 2;
    }
    const uint32_t    max_trace_depth  = 1;
    
    OptixProgramGroup program_groups[] = { 
        raygen_prog_group, 
        miss_prog_groups[0], 
        miss_prog_groups[1], 
        hitgroup_prog_groups[0],
        hitgroup_prog_groups[1]
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
                map->context,
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



    // 4. setup shader binding table
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    
    // Number of ray types: for normals, ranges -> 2
    unsigned int ray_type_count = 2;

    CUdeviceptr d_miss_records;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size * ray_type_count ) );
    
    MissSbtRecord ms_sbt[2];
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_groups[0], &ms_sbt[0] ) );
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_groups[1], &ms_sbt[1] ) );


    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss_records ),
                ms_sbt,
                miss_record_size * ray_type_count,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr d_hitgroup_records;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_hitgroup_records ), hitgroup_record_size * ray_type_count ) );
    
    HitGroupSbtRecord hg_sbt[2];

    for(unsigned int ray_id = 0; ray_id < 2; ray_id++)
    {
        OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_groups[ray_id], &hg_sbt[ray_id] ) );

        Memory<Vector*, RAM> normals_cpu(map->meshes.size());
        for(size_t i=0; i<map->meshes.size(); i++)
        {
            normals_cpu[i] = map->meshes[i].normals.raw();
        }
        
        cudaMalloc(reinterpret_cast<void**>(&hg_sbt[ray_id].data.normals), map->meshes.size() * sizeof(Vector*));
        // gpu array of gpu pointers
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(hg_sbt[ray_id].data.normals),
                    reinterpret_cast<void*>(normals_cpu.raw()),
                    map->meshes.size() * sizeof(Vector*),
                    cudaMemcpyHostToDevice
                    ) );
    }

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_hitgroup_records ),
                hg_sbt,
                hitgroup_record_size * ray_type_count,
                cudaMemcpyHostToDevice
                ) );

    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = d_miss_records;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = ray_type_count;
    sbt.hitgroupRecordBase          = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = ray_type_count;
}

} // namespace imagine