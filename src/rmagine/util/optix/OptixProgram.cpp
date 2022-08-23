#include "rmagine/util/optix/OptixProgram.hpp"

#include "rmagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace rmagine {

RayGenModule::~RayGenModule()
{
    if(record_h)
    {
        cudaFreeHost(record_h);
    }

    if(record)
    {
        cudaFree( reinterpret_cast<void*>( record ) );
    }

    if(prog_group)
    {
        optixProgramGroupDestroy( prog_group );
    }

    if(module)
    {
        optixModuleDestroy( module );
    }
}


HitModule::~HitModule()
{
    if(record_miss_h)
    {
        cudaFreeHost(record_miss_h);
    }

    if(record_miss)
    {
        cudaFree( reinterpret_cast<void*>( record_miss ) );
    }

    if(record_hit_h)
    {
        cudaFreeHost(record_hit_h);
    }

    if(record_hit)
    {
        cudaFree( reinterpret_cast<void*>( record_hit ) );
    }

    if(prog_group_hit)
    {
        optixProgramGroupDestroy( prog_group_hit );
    }

    if(prog_group_miss)
    {
        optixProgramGroupDestroy( prog_group_miss );
    }

    if(module)
    {
        optixModuleDestroy( module );
    }
}


OptixSBT::~OptixSBT()
{
    // if(sbt.raygenRecord)
    // {
    //     cudaFree( reinterpret_cast<void*>( sbt.raygenRecord ) );
    // }

    // if(sbt.missRecordBase)
    // {
    //     cudaFree( reinterpret_cast<void*>( sbt.missRecordBase ) );
    // }

    // if(sbt.hitgroupRecordBase)
    // {
    //     cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) );
    // }
}


OptixSensorPipeline::~OptixSensorPipeline()
{
    if(pipeline)
    {
        optixPipelineDestroy( pipeline );
    }
}


OptixProgram::~OptixProgram()
{
    // std::cout << "Destruct OptixProgram" << std::endl;
    cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) );
    cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) );
    cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) );

    optixPipelineDestroy( pipeline );
    optixProgramGroupDestroy( hitgroup_prog_group );
    optixProgramGroupDestroy( miss_prog_group );
    optixProgramGroupDestroy( raygen_prog_group );
    optixModuleDestroy( module );
}

} // namespace rmagine