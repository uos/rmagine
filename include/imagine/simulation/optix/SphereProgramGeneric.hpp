#ifndef IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>

#include "imagine/util/optix/OptixSbtRecord.hpp"

namespace imagine {

typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;

class SphereProgramGeneric : public OptixProgram
{
public:
    SphereProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericSphere& flags);

    ~SphereProgramGeneric();

private:
    HitGroupSbtRecord m_hg_sbt;
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP