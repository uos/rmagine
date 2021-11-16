#include "imagine/util/optix/OptixProgram.hpp"

#include "imagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace imagine {

OptixProgram::~OptixProgram()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );

    OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( module ) );   
}


} // namespace imagine