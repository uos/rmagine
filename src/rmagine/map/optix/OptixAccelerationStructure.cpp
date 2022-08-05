#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include <cuda_runtime.h>

#include "rmagine/util/optix/OptixDebug.hpp"

namespace rmagine
{

OptixAccelerationStructure::~OptixAccelerationStructure()
{
    if(buffer)
    {
        cudaFree( reinterpret_cast<void*>( buffer ) );
    }
}

} // namespace rmagine