#ifndef IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>

namespace imagine {

class SphereProgramGeneric : public OptixProgram
{
public:
    SphereProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericSphere& flags);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP