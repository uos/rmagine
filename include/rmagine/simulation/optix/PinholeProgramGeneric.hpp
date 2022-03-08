#ifndef RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

namespace rmagine {

class PinholeProgramGeneric : public OptixProgram
{
public:
    PinholeProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericPinhole& flags);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP