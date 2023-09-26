#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include <cuda_runtime.h>

#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix_types.h>
#include <cassert>

namespace rmagine
{

static_assert(sizeof(unsigned long long) == sizeof(OptixTraversableHandle), "OptixTraversableHandle is not unsigned long long");

OptixAccelerationStructure::~OptixAccelerationStructure()
{
    if(buffer)
    {
        // std::cout << "[OptixAccelerationStructure::~OptixAccelerationStructure()] free " << buffer_size / 1000000 << " MB" << std::endl;
        cudaFree( reinterpret_cast<void*>( buffer ) );
        // std::cout << "[OptixAccelerationStructure::~OptixAccelerationStructure()] done." << std::endl;
    }
    
}

} // namespace rmagine