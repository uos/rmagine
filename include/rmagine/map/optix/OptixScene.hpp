#ifndef RMAGINE_MAP_OPTIX_SCENE_HPP
#define RMAGINE_MAP_OPTIX_SCENE_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/IDGen.hpp>

#include "optix_definitions.h"

#include "OptixEntity.hpp"

namespace rmagine
{

class OptixScene 
: public OptixEntity
{
public:
    OptixScene(OptixContextPtr context = optix_default_context());
    OptixScene(OptixGeometryPtr geom, OptixContextPtr context = optix_default_context());
    
    void set(OptixGeometryPtr geom);
private:
    OptixGeometryPtr m_geom;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP