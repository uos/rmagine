#ifndef RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP
#define RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP

#include <cuda_runtime.h>
#include <memory>
#include <cuda.h> // CUdeviceptr

namespace rmagine
{

class OptixAccelerationStructure
{
public:
    ~OptixAccelerationStructure();

    unsigned long long          handle;
    CUdeviceptr                 buffer = 0;
    size_t                      buffer_size = 0;
    size_t                      n_elements = 0;

    
};

using OptixAccelerationStructurePtr = std::shared_ptr<OptixAccelerationStructure>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP