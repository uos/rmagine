#ifndef RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/map/optix/optix_definitions.h>
#include <rmagine/map/optix/optix_sbt.h>

#include <memory>

namespace rmagine {

class OnDnProgramGeneric : public OptixProgram
{
    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;
public:
    OnDnProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericOnDn& flags);

    OnDnProgramGeneric(
        OptixScenePtr scene,
        const OptixSimulationDataGenericOnDn& flags);

    virtual ~OnDnProgramGeneric();

    void updateSBT();

private:
    // currently used scene
    OptixScenePtr       m_scene;

    Memory<RayGenSbtRecord, RAM> rg_sbt;
    Memory<MissSbtRecord, RAM> ms_sbt;
    Memory<HitGroupSbtRecord, RAM> hg_sbt;
};

using OnDnProgramGenericPtr = std::shared_ptr<OnDnProgramGeneric>;

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_GENERIC_HPP