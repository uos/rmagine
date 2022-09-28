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

    // virtual void commit() = 0;

    virtual OptixGeometryType type() const = 0;

    // if child -> 0, else max of child + 1
    virtual unsigned int depth() const = 0;

    // virtual void apply() = 0 in OptixTransformable
    virtual void commit() = 0;

    // handle parents
    void cleanupParents();
    std::unordered_set<OptixScenePtr> parents() const;
    bool removeParent(OptixScenePtr parent);
    bool hasParent(OptixScenePtr parent) const;
    void addParent(OptixScenePtr parent);

    OptixScenePtr makeScene();
    OptixInstPtr instantiate();

protected:
    std::unordered_set<OptixSceneWPtr> m_parents;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_GEOMETRY_HPP

