#ifndef RMAGINE_MAP_OPTIX_SPHERES_HPP
#define RMAGINE_MAP_OPTIX_SPHERES_HPP

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

class OptixSpheres : public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixSpheres(OptixContextPtr context = optix_default_context());

    virtual ~OptixSpheres();

    virtual void apply();
    virtual void commit();

    Memory<Point, VRAM_CUDA> centers;
    Memory<float, VRAM_CUDA> radii;

    Memory<Point, VRAM_CUDA> centers_;
    Memory<float, VRAM_CUDA> radii_;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SPHERES_HPP