#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP

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

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP