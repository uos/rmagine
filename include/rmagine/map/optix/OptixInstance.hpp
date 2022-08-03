#ifndef RMAGINE_MAP_OPTIX_INSTANCE_HPP
#define RMAGINE_MAP_OPTIX_INSTANCE_HPP

#include "OptixGeometry.hpp"
#include <rmagine/util/optix/OptixContext.hpp>

#include <optix_types.h>

namespace rmagine
{

class OptixInst : public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixInst(OptixGeometryPtr geom, OptixContextPtr context = optix_default_context());

    virtual ~OptixInst();

    virtual void apply();

    void setId(unsigned int id);

    unsigned int id() const;

    OptixInstance data() const;

    virtual void commit();
protected:
    OptixInstance m_data;
    OptixGeometryPtr m_geom;
};

using OptixInstPtr = std::shared_ptr<OptixInst>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCE_HPP