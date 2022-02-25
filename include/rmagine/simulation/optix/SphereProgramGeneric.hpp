#ifndef IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include "rmagine/util/optix/OptixSbtRecord.hpp"

namespace rmagine {

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

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP