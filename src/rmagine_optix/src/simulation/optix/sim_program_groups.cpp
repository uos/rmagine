#include "rmagine/simulation/optix/sim_program_groups.h"
#include "rmagine/simulation/optix/sim_modules.h"

#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix_stubs.h>

namespace rmagine 
{

SimRayGenProgramGroup::~SimRayGenProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
}

SimMissProgramGroup::~SimMissProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
}

SimHitProgramGroup::~SimHitProgramGroup()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }
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
                // std::cout << "HitGroup update number of records: " << record_count << " -> " << n_hitgroups_required << std::endl;
                if(record_h)
                {
                    RM_CUDA_CHECK( cudaFreeHost( record_h ) );
                }
                
                RM_CUDA_CHECK( cudaMallocHost( &record_h, n_hitgroups_required * record_stride ) );

                for(size_t i=0; i<n_hitgroups_required; i++)
                {
                    RM_OPTIX_CHECK( optixSbtRecordPackHeader( prog_group, &record_h[i] ) );
                }

                if( record )
                {
                    RM_CUDA_CHECK( cudaFree( reinterpret_cast<void*>( record ) ) );
                }
                
                RM_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &record ), n_hitgroups_required * record_stride ) );

                record_count = n_hitgroups_required;
            }
        }

        for(size_t i=0; i<record_count; i++)
        {
            record_h[i].data = scene->sbt_data;
        }

        RM_CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( record ),
                    record_h,
                    record_count * record_stride,
                    cudaMemcpyHostToDevice,
                    scene->stream()->handle()
                    ) );
    }
    
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

    // set description
    ret->description->kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    ret->description->raygen.module = module->module;
    ret->description->raygen.entryFunctionName = "__raygen__rg";

    // set options
    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif

    // set module
    ret->module = module;

    ret->create();

    { // init SBT Records
        const size_t raygen_record_size     = sizeof( SimRayGenProgramGroup::SbtRecordData );
        
        RM_CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            raygen_record_size ) );

        RM_OPTIX_CHECK( optixSbtRecordPackHeader( 
            ret->prog_group,
            &ret->record_h[0] ) );

        RM_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), raygen_record_size ) );

        RM_CUDA_CHECK( cudaMemcpyAsync(
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

    ret->description->kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ret->description->raygen.module            = module->module;
    ret->description->raygen.entryFunctionName = "__miss__ms";

    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif
    ret->module = module;

    ret->create();

    { // init SBT Records
        const size_t miss_record_size     = sizeof( SimMissProgramGroup::SbtRecordData );
        
        RM_CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            miss_record_size ) );

        RM_OPTIX_CHECK( optixSbtRecordPackHeader( 
            ret->prog_group,
            &ret->record_h[0] ) );

        RM_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), miss_record_size ) );

        RM_CUDA_CHECK( cudaMemcpyAsync(
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

    ret->description->kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    ret->description->raygen.module            = module->module;
    ret->description->raygen.entryFunctionName = "__closesthit__ch";

    #if OPTIX_VERSION >= 70400
    ret->options->payloadType = &module->compile_options->payloadTypes[0];
    #endif
    ret->module = module;

    ret->create();

    { // init SBT Records
        const size_t n_hitgroup_records = scene->requiredSBTEntries();   
        const size_t hitgroup_record_size     = sizeof( SimMissProgramGroup::SbtRecordData ) * n_hitgroup_records;
        
        RM_CUDA_CHECK( cudaMallocHost( 
            &ret->record_h, 
            hitgroup_record_size ) );

        for(size_t i=0; i<n_hitgroup_records; i++)
        {
            RM_OPTIX_CHECK( optixSbtRecordPackHeader( 
                ret->prog_group,
                &ret->record_h[i] ) );
            ret->record_h[i].data = scene->sbt_data;
        }
        
        RM_CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ret->record ), hitgroup_record_size ) );

        RM_CUDA_CHECK( cudaMemcpyAsync(
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


} // namespace rmagine