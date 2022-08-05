#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include "rmagine/util/optix/OptixSbtRecord.hpp"

#include <rmagine/map/optix/optix_definitions.h>

namespace rmagine {

// typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;
typedef SbtRecord<HitGroupDataMesh>      HitGroupSbtRecordMesh;

class SphereProgramGeneric : public OptixProgram
{
public:
    SphereProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericSphere& flags);

    SphereProgramGeneric(
        OptixScenePtr scene,
        const OptixSimulationDataGenericSphere& flags
    );

    ~SphereProgramGeneric();

private:
    HitGroupSbtRecordMesh hg_sbt;
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP