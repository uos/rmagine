#include "rmagine/simulation/optix/PinholeProgramNormals.hpp"

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

PinholeProgramNormals::PinholeProgramNormals(OptixMapPtr map)
{
    const char *kernel =
    #include "kernels/PinholeProgramNormalsString.h"
    ;

    OptixScenePtr scene = map->scene();

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;

    OptixGeometryPtr geom = scene->getRoot();
    OptixInstancesPtr insts = std::dynamic_pointer_cast<OptixInstances>(geom);

    if(insts)
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

    OPTIX_CHECK( optixModuleCreateFromPTX(
                map->context()->ref(),
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ));

    // 2. initProgramGroups
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros
    sizeof_log = sizeof( log );

    // 2.1 Raygen
    {
        OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    map->context()->ref(),
                    &raygen_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &raygen_prog_group
                    ) );
    }

    // 2.2 Miss program
    {
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    map->context()->ref(),
                    &miss_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &miss_prog_group
                    ) );
    }

    // 2.3 Closest Hit program
    {
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    map->context()->ref(),
                    &hitgroup_prog_group_desc,
                    1,   // num program groups
                    &program_group_options,
                    log,
                    &sizeof_log,
                    &hitgroup_prog_group
                    ) );
    }

    // 3. link pipeline
    // traverse depth = 2 for ias + gas
    uint32_t    max_traversable_depth = 1;
    if(insts)
    {
        max_traversable_depth = 2;
    }
    const uint32_t    max_trace_depth  = 1;
    
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
                map->context()->ref(),
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


PinholeProgramNormals::~PinholeProgramNormals()
{
    // std::cout << "Destruct SphereProgramNormals" << std::endl;
    // cudaFree(m_hg_sbt.data.normals);
}

void PinholeProgramNormals::updateSBT()
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