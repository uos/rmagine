#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/map/optix/optix_definitions.h>

#include <memory>

namespace rmagine {

class SphereProgramGeneric : public OptixProgram
{
    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;

public:
    SphereProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericSphere& flags);

    SphereProgramGeneric(
        OptixScenePtr scene,
        const OptixSimulationDataGenericSphere& flags
    );

    virtual ~SphereProgramGeneric();

    void updateSBT();

private:
    // scene container
    OptixMapPtr         m_map;
    // currently used scene
    OptixScenePtr       m_scene;


    Memory<RayGenSbtRecord, RAM> rg_sbt;
    Memory<MissSbtRecord, RAM> ms_sbt;
    Memory<HitGroupSbtRecord, RAM> hg_sbt;
};

using SphereProgramGenericPtr = std::shared_ptr<SphereProgramGeneric>;

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_GENERIC_HPP