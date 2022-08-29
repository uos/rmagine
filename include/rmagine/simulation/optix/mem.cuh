#ifndef RMAGINE_SIMULATION_OPTIX_MEM_H
#define RMAGINE_SIMULATION_OPTIX_MEM_H

#include <rmagine/simulation/optix/sim_program_data.h>

extern "C" {
__constant__ rmagine::OptixSimulationDataGeneric mem;
}

#endif // RMAGINE_SIMULATION_OPTIX_MEM_H