#ifndef RMAGINE_MAP_OPTIX_INSTANCE_HPP
#define RMAGINE_MAP_OPTIX_INSTANCE_HPP


#include <optix_types.h>

#include "optix_definitions.h"

#include "OptixScene.hpp"
#include "OptixTransformable.hpp"

// #include <optix_types.h>

// #include <cuda.h>
// #include <cuda_runtime.h>

namespace rmagine
{

class OptixInst 
: public OptixEntity
, public OptixTransformable
{
public:
    using Base = OptixGeometry;

    OptixInst(OptixContextPtr context = optix_default_context());

    OptixInst(OptixGeometryPtr geom);

    virtual ~OptixInst();

    void setGeometry(OptixGeometryPtr geom);
    OptixGeometryPtr geometry() const;

    virtual void apply();

    void setId(unsigned int id);

    unsigned int id() const;

    void disable();

    void enable();

    OptixInstance data() const;
    CUdeviceptr data_gpu() const;

protected:
    
    OptixInstance m_data;
    // filled after commit
    CUdeviceptr m_data_gpu = 0;
    // CUdeviceptr m_data_gpu;

    OptixGeometryPtr m_geom;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCE_HPP