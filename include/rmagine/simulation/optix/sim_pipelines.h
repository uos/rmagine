#ifndef RMAGINE_SIMULATION_OPTIX_SIM_PIPELINES_H
#define RMAGINE_SIMULATION_OPTIX_SIM_PIPELINES_H

// rmagine optix module interface
#include <rmagine/util/optix/optix_modules.h>

// map connection
#include <rmagine/map/optix/OptixSceneEventReceiver.hpp>

// sensor connection
#include "sim_program_data.h"

namespace rmagine
{

struct SimPipeline 
: public Pipeline
, public OptixSceneEventReceiver
{
    virtual void onDepthChanged() override;

    virtual void onCommitDone(const OptixSceneCommitResult& info) override;

    virtual ~SimPipeline();
};

using SimPipelinePtr = std::shared_ptr<SimPipeline>;

// Pipeline
SimPipelinePtr make_pipeline_sim(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags);

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SIM_PIPELINES_H