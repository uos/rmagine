#ifndef RMAGINE_MAP_OPTIX_SHAPES_H
#define RMAGINE_MAP_OPTIX_SHAPES_H

#include "optix_definitions.h"
#include "OptixMesh.hpp"

namespace rmagine
{

class OptixSphere : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixSphere(unsigned int num_long = 50,
        unsigned int num_lat = 50,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixSphere();
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SHAPES_H