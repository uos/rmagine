#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

#include <rmagine/map/optix/optix_definitions.h>

namespace rmagine {

class SphereProgramRanges : public OptixProgram
{
public:
    // deprecated
    SphereProgramRanges(OptixMapPtr map);

    SphereProgramRanges(OptixGeometryPtr geom);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP