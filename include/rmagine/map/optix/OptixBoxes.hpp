#ifndef RMAGINE_MAP_OPTIX_BOXES_HPP
#define RMAGINE_MAP_OPTIX_BOXES_HPP

#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <rmagine/util/cuda/CudaContext.hpp>
#include <rmagine/util/optix/OptixContext.hpp>

#include <memory>

#include <assimp/mesh.h>

#include "OptixGeometry.hpp"

#include "optix_definitions.h"

namespace rmagine
{

/// WIP: BOXES AS CUSTOM PRIMITIVES
class OptixBoxes : public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixBoxes(OptixContextPtr context = optix_default_context());

    virtual ~OptixBoxes();

    virtual void apply();
    virtual void commit();

    Memory<OptixAabb, VRAM_CUDA> boxes;
};

using OptixBoxesPtr = std::shared_ptr<OptixBoxes>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_BOXES_HPP