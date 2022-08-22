#ifndef RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/map/optix/optix_definitions.h>
#include <rmagine/map/optix/optix_sbt.h>

#include <memory>


namespace rmagine {

class O1DnProgramGeneric : public OptixProgram
{
    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;

public:
    O1DnProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericO1Dn& flags);

    O1DnProgramGeneric(
        OptixScenePtr scene,
        const OptixSimulationDataGenericO1Dn& flags);

    virtual ~O1DnProgramGeneric();

    void updateSBT();

private:
    // currently used scene
    OptixScenePtr       m_scene;

    Memory<RayGenSbtRecord, RAM> rg_sbt;
    Memory<MissSbtRecord, RAM> ms_sbt;
    Memory<HitGroupSbtRecord, RAM> hg_sbt;
};

using O1DnProgramGenericPtr = std::shared_ptr<O1DnProgramGeneric>;

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_GENERIC_HPP