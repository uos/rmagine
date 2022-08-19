#ifndef RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP
#define RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP

#include <optix_types.h>
#include <cuda_runtime.h>
#include <memory>

namespace rmagine
{

class OptixAccelerationStructure
{
public:

    ~OptixAccelerationStructure();

    OptixTraversableHandle      handle;
    CUdeviceptr                 buffer = 0;
    size_t                      buffer_size = 0;
    size_t                      n_elements = 0;

    
};

using OptixAccelerationStructurePtr = std::shared_ptr<OptixAccelerationStructure>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_ACCELERATION_STRUCTURE_HPP