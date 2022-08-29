#ifndef RMAGINE_SIMULATION_OPTIX_SIM_PGS_H
#define RMAGINE_SIMULATION_OPTIX_SIM_PGS_H

// #include <optix.h>

// rmagine optix module interface
#include <rmagine/util/optix/optix_modules.h>
#include <rmagine/util/optix/OptixData.hpp>
#include <rmagine/util/optix/OptixSbtRecord.hpp>

// map connection
#include <rmagine/map/optix/OptixSceneEventReceiver.hpp>
#include <rmagine/map/optix/optix_sbt.h>

// sensor connection
#include "sim_program_data.h"

namespace rmagine
{

struct SimRayGenProgramGroup 
: public ProgramGroup
{
    using RecordData        = RayGenDataEmpty;
    using SbtRecordData     = SbtRecord<RecordData>;

    SbtRecordData*        record_h      = nullptr;

    virtual ~SimRayGenProgramGroup();
};

using SimRayGenProgramGroupPtr = std::shared_ptr<SimRayGenProgramGroup>;


struct SimMissProgramGroup
: public ProgramGroup
{
    using RecordData          = MissDataEmpty;
    using SbtRecordData       = SbtRecord<RecordData>;

    SbtRecordData*        record_h      = nullptr;

    virtual ~SimMissProgramGroup();
};

using SimMissProgramGroupPtr = std::shared_ptr<SimMissProgramGroup>;

struct SimHitProgramGroup
: public ProgramGroup
, public OptixSceneEventReceiver
{
    using RecordData        = OptixSceneSBT;
    using SbtRecordData     = SbtRecord<RecordData>;

    SbtRecordData*        record_h      = nullptr;

    virtual void onSBTUpdated(bool size_changed) override;

    virtual ~SimHitProgramGroup();
};

using SimHitProgramGroupPtr = std::shared_ptr<SimHitProgramGroup>;



/// ProgramGroup
// - RayGen
SimRayGenProgramGroupPtr make_program_group_sim_gen(
    OptixScenePtr scene,
    ProgramModulePtr module);

SimRayGenProgramGroupPtr make_program_group_sim_gen(
    OptixScenePtr scene,
    unsigned int sensor_id);

// - Miss
SimMissProgramGroupPtr make_program_group_sim_miss(
    OptixScenePtr scene,
    ProgramModulePtr module);

SimMissProgramGroupPtr make_program_group_sim_miss(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags);

// - Hit
SimHitProgramGroupPtr make_program_group_sim_hit(
    OptixScenePtr scene,
    ProgramModulePtr module);

SimHitProgramGroupPtr make_program_group_sim_hit(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags);


} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SIM_PGS_H