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

#include <map>

namespace rmagine {

// typedef SbtRecord<RayGenDataEmpty>     RayGenSbtRecord;
// typedef SbtRecord<MissDataEmpty>       MissSbtRecord;

typedef SbtRecord<RayGenDataEmpty>     RayGenSbtRecord;
typedef SbtRecord<MissDataEmpty>       MissSbtRecord;
// typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;


SphereProgramGeneric::SphereProgramGeneric(
    OptixMapPtr map,
    const OptixSimulationDataGenericSphere& flags)
{
    std::cout << "Construct SphereProgramGeneric" << std::endl;

    const char *kernel =
    #include "kernels/SphereProgramGenericString.h"
    ;

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    
    OptixModuleCompileBoundValueEntry options[6];
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
    // computeObjectIds
    options[5] = {};
    options[5].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeObjectIds);
    options[5].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeObjectIds );
    options[5].boundValuePtr = &flags.computeObjectIds;


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
    module_compile_options.numBoundValues = 6;
    
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

    // 2.1. RAYGEN
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

    // 2.2 Miss programs
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};

        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
                map->context()->ref(),
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
        // hitgroup_prog_group_desc.hitgroup.moduleAH            = nullptr;
        // hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        // hitgroup_prog_group_desc.hitgroup.moduleIS            = nullptr;
        // hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;

        // 
        // hitgroup_prog_group_desc.hitgroup.moduleIS            = sphere_module;
        // hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__box";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                map->context()->ref(),
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
    uint32_t    max_traversable_depth = 1;
    if(map->ias())
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

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size) );
    
    MissSbtRecord ms_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );


    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecordMesh );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    

    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );

    { // inst to mesh
        Memory<unsigned int, RAM> inst_to_mesh(map->meshes.size());
        for(size_t i=0; i<map->meshes.size(); i++)
        {
            inst_to_mesh[i] = i;
        }

        cudaMalloc(reinterpret_cast<void**>(&hg_sbt.data.inst_to_mesh), inst_to_mesh.size() * sizeof(unsigned int));
        // gpu array of gpu pointers
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(hg_sbt.data.inst_to_mesh),
                    reinterpret_cast<void*>(inst_to_mesh.raw()),
                    inst_to_mesh.size() * sizeof(unsigned int),
                    cudaMemcpyHostToDevice
                    ) );
        
    }

    { // normals
        Memory<Vector*, RAM> normals_cpu(map->meshes.size());
        for(size_t i=0; i<map->meshes.size(); i++)
        {
            normals_cpu[i] = map->meshes[i].face_normals.raw();
        }
        
        cudaMalloc(reinterpret_cast<void**>(&hg_sbt.data.normals), map->meshes.size() * sizeof(Vector*));
        // gpu array of gpu pointers
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(hg_sbt.data.normals),
                    reinterpret_cast<void*>(normals_cpu.raw()),
                    map->meshes.size() * sizeof(Vector*),
                    cudaMemcpyHostToDevice
                    ) );
    } 

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );

    // std::cout << "Free hg_sbt.data.normals" << std::endl;
    // cudaFree(hg_sbt.data.normals);
    // TODO shader binding table to map?

    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecordMesh );
    sbt.hitgroupRecordCount         = 1;
}


SphereProgramGeneric::SphereProgramGeneric(
    OptixScenePtr scene,
    const OptixSimulationDataGenericSphere& flags)
{
    std::cout << "Construct SphereProgramGeneric on Scene" << std::endl;

    const char *kernel =
    #include "kernels/SphereProgramGenericString.h"
    ;

    // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );
    
    OptixModuleCompileBoundValueEntry options[6];
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
    // computeObjectIds
    options[5] = {};
    options[5].pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGenericSphere, computeObjectIds);
    options[5].sizeInBytes = sizeof( OptixSimulationDataGenericSphere::computeObjectIds );
    options[5].boundValuePtr = &flags.computeObjectIds;


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
    module_compile_options.numBoundValues = 6;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;

    std::cout << "Scene get root" << std::endl;
    OptixGeometryPtr geom = scene->getRoot();

    OptixInstancesPtr insts = std::dynamic_pointer_cast<OptixInstances>(geom);
    if(insts)
    {
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    } else {
        std::cout << "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS" << std::endl;
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
                scene->context()->ref(),
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ));

    std::cout << "Compiled Generic Shader" << std::endl;

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
                    geom->context()->ref(),
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
                geom->context()->ref(),
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


    std::cout << "Construct SBT ..." << std::endl;
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

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size) );
    
    MissSbtRecord ms_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );


    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecordMesh );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    
    // HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );

    { // connections
        Memory<unsigned int, RAM> inst_to_mesh;

        if(insts)
        {
            // instances are available
            auto instances = insts->instances();
            size_t Ninstances = instances.rbegin()->first + 1;

            inst_to_mesh.resize(Ninstances);
            for(unsigned int i=0; i<inst_to_mesh.size(); i++)
            {
                inst_to_mesh[i] = -1;
            }

            for(auto elem : instances)
            {
                unsigned int inst_id = elem.first;
                OptixGeometryPtr geom = elem.second->geometry();
                OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(geom);

                if(mesh)
                {
                    unsigned int mesh_id = scene->get(mesh);
                    inst_to_mesh[inst_id] = mesh_id;
                }
            }
        } else {
            // only one mesh 0 -> 0
            inst_to_mesh.resize(1);
            inst_to_mesh[0] = 0;
        }

        cudaMalloc(reinterpret_cast<void**>(&hg_sbt.data.inst_to_mesh), inst_to_mesh.size() * sizeof(unsigned int));
        // gpu array of gpu pointers
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(hg_sbt.data.inst_to_mesh),
                    reinterpret_cast<void*>(inst_to_mesh.raw()),
                    inst_to_mesh.size() * sizeof(unsigned int),
                    cudaMemcpyHostToDevice
                    ) );
    }
    
    {
        std::map<unsigned int, OptixGeometryPtr> geoms = scene->geometries();

        // get last id
        unsigned int normal_buffer_size = geoms.rbegin()->first + 1;

        std::cout << "Meshes: " << normal_buffer_size << std::endl; 

        Memory<Vector*, RAM> normals_cpu(normal_buffer_size);
        for(auto elem : geoms)
        {
            // check if mesh
            OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(elem.second);
            if(mesh)
            {
                normals_cpu[elem.first] = mesh->face_normals.raw();
            } else {
                std::cout << "NO MESH: how to handle normals?" << std::endl;
            }
        }
        
        cudaMalloc(reinterpret_cast<void**>(&hg_sbt.data.normals), normals_cpu.size() * sizeof(Vector*));
        // gpu array of gpu pointers
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>(hg_sbt.data.normals),
                    reinterpret_cast<void*>(normals_cpu.raw()),
                    normals_cpu.size() * sizeof(Vector*),
                    cudaMemcpyHostToDevice
                    ) );
    }
    
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );

    // std::cout << "Free hg_sbt.data.normals" << std::endl;
    // cudaFree(hg_sbt.data.normals);
    // TODO shader binding table to map?

    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecordMesh );
    sbt.hitgroupRecordCount         = 1;
}

SphereProgramGeneric::~SphereProgramGeneric()
{
    // std::cout << "Destruct SphereProgramGeneric" << std::endl;
    // cudaFree(hg_sbt.data.normals);
    cudaFree(hg_sbt.data.inst_to_mesh);
    cudaFree(hg_sbt.data.normals);
}


} // namespace rmagine