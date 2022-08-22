#ifndef RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP
#define RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/map/optix/optix_definitions.h>
#include <rmagine/map/optix/optix_sbt.h>

#include <memory>

namespace rmagine {

class PinholeProgramGeneric : public OptixProgram
{
    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;
public:
    PinholeProgramGeneric(
        OptixMapPtr map,
        const OptixSimulationDataGenericPinhole& flags);

    PinholeProgramGeneric(
        OptixScenePtr scene,
        const OptixSimulationDataGenericPinhole& flags);

    virtual ~PinholeProgramGeneric();

    void updateSBT();

private:
    // currently used scene
    OptixScenePtr       m_scene;

    Memory<RayGenSbtRecord, RAM> rg_sbt;
    Memory<MissSbtRecord, RAM> ms_sbt;
    Memory<HitGroupSbtRecord, RAM> hg_sbt;
};

using PinholeProgramGenericPtr = std::shared_ptr<PinholeProgramGeneric>;

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_GENERIC_HPP