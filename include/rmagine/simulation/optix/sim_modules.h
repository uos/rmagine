#ifndef RMAGINE_SIMULATION_OPTIX_SIM_MODULES_H
#define RMAGINE_SIMULATION_OPTIX_SIM_MODULES_H


// rmagine optix module interface
#include <rmagine/util/optix/optix_modules.h>

// map connection
#include <rmagine/map/optix/optix_definitions.h>

// sensor connection
#include "OptixSimulationData.hpp"

namespace rmagine
{

// ProgramModule
// - Gen
ProgramModulePtr make_program_module_sim_gen(
    OptixScenePtr scene,
    unsigned int sensor_id);

// - Hit, Miss
ProgramModulePtr make_program_module_sim_hit_miss(
    OptixScenePtr scene,
    const OptixSimulationDataGeneric& flags);

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SIM_MODULES_H