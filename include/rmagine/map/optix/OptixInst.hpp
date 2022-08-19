#ifndef RMAGINE_MAP_OPTIX_INSTANCE_HPP
#define RMAGINE_MAP_OPTIX_INSTANCE_HPP


#include <optix_types.h>

#include "optix_definitions.h"

#include "OptixGeometry.hpp"
#include "OptixScene.hpp"

// #include <optix_types.h>

// #include <cuda.h>
// #include <cuda_runtime.h>

namespace rmagine
{

class OptixInst 
: public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixInst(OptixContextPtr context = optix_default_context());

    virtual ~OptixInst();

    void set(OptixScenePtr geom);
    OptixScenePtr scene() const;

    virtual void apply();
    // virtual void commit();
    virtual unsigned int depth() const;

    void setId(unsigned int id);
    unsigned int id() const;

    void disable();
    void enable();

    virtual OptixGeometryType type() const
    {
        return OptixGeometryType::INSTANCE;
    }

    OptixInstance data() const;
    CUdeviceptr data_gpu() const;
protected:
    
    OptixInstance m_data;
    // filled after commit
    CUdeviceptr m_data_gpu = 0;
    // CUdeviceptr m_data_gpu;

    OptixScenePtr m_scene;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCE_HPP