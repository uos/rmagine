#ifndef IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_RANGES_HPP
#define IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_RANGES_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class OnDnProgramRanges : public OptixProgram
{
public:
    OnDnProgramRanges(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_RANGES_HPP