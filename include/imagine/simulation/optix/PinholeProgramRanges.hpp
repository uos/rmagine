#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class PinholeProgramRanges : public OptixProgram
{
public:
    PinholeProgramRanges(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_RANGES_HPP