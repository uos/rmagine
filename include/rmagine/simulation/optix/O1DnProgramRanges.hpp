#ifndef RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP
#define RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class O1DnProgramRanges : public OptixProgram
{
public:
    O1DnProgramRanges(OptixMapPtr map);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP