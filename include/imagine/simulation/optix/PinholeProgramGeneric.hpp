#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>

namespace imagine {

class PinholeProgramGeneric : public OptixProgram
{
public:
    PinholeProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericPinhole& flags);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP