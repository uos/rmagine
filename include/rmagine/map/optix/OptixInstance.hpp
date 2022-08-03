#ifndef RMAGINE_MAP_OPTIX_INSTANCE_HPP
#define RMAGINE_MAP_OPTIX_INSTANCE_HPP

#include "OptixGeometry.hpp"
#include <rmagine/util/optix/OptixContext.hpp>

#include <optix_types.h>

namespace optix
{
    // using OptixInstance = ::OptixInstace;
    using OptixInstancePtr = std::shared_ptr<OptixInstance>;
}

namespace rmagine
{

class OptixInstance : public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixInstance(OptixGeometryPtr geom, OptixContextPtr context = optix_default_context());

    virtual ~OptixInstance();

    virtual void apply();

    virtual void commit();
protected:
    ::OptixInstance m_instance;
    OptixGeometryPtr m_geom;
};

using OptixInstancePtr = std::shared_ptr<OptixInstance>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCE_HPP