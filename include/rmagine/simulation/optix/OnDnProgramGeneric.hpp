#ifndef RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

namespace rmagine {

class OnDnProgramGeneric : public OptixProgram
{
public:
    OnDnProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericOnDn& flags);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP