#ifndef IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>

namespace imagine {

class O1DnProgramGeneric : public OptixProgram
{
public:
    O1DnProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericO1Dn& flags);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP