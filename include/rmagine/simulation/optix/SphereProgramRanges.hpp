#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class SphereProgramRanges : public OptixProgram
{
public:
    SphereProgramRanges(OptixMapPtr map);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_RANGES_HPP