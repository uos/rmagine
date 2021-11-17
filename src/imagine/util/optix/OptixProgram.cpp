#include "imagine/util/optix/OptixProgram.hpp"

#include "imagine/util/optix/OptixDebug.hpp"
#include <optix_stubs.h>

#include <cuda_runtime.h>

namespace imagine {

OptixProgram::~OptixProgram()
{
    cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) );
    cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) );
    cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) );

    optixPipelineDestroy( pipeline );
    optixProgramGroupDestroy( hitgroup_prog_group );
    optixProgramGroupDestroy( miss_prog_group );
    optixProgramGroupDestroy( raygen_prog_group );
    optixModuleDestroy( module );
}

} // namespace imagine