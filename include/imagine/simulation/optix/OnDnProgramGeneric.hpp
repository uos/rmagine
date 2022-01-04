#ifndef IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>

namespace imagine {

class OnDnProgramGeneric : public OptixProgram
{
public:
    OnDnProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericOnDn& flags);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP