#ifndef RMAGINE_MAP_OPTIX_GEOMETRY_HPP
#define RMAGINE_MAP_OPTIX_GEOMETRY_HPP

#include <memory>
#include <rmagine/math/types.h>
#include <rmagine/util/optix/OptixContext.hpp>

#include <unordered_set>


#include "optix_definitions.h"
#include "OptixEntity.hpp"
#include "OptixTransformable.hpp"


namespace rmagine
{

class OptixGeometry
: public OptixEntity
, public OptixTransformable
{
public:
    OptixGeometry(OptixContextPtr context = optix_default_context());

    virtual ~OptixGeometry();

    OptixAccelerationStructurePtr acc();

    virtual void commit() = 0;

    // geometry can be instanced
    void cleanupParents();
    std::unordered_set<OptixInstPtr> parents() const;
    void addParent(OptixInstPtr parent);

protected:
    std::unordered_set<OptixInstWPtr> m_parents;
    OptixAccelerationStructurePtr m_as;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_GEOMETRY_HPP

