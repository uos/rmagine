#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class PinholeProgramRanges : public OptixProgram
{
public:
    PinholeProgramRanges(OptixMapPtr map);
};

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP