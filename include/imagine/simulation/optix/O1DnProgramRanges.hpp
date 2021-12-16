#ifndef IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP
#define IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class O1DnProgramRanges : public OptixProgram
{
public:
    O1DnProgramRanges(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_RANGES_HPP