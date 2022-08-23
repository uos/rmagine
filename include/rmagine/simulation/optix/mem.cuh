#ifndef RMAGINE_SIMULATION_OPTIX_MEM_H
#define RMAGINE_SIMULATION_OPTIX_MEM_H

#include <optix.h>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

extern "C" {
__constant__ rmagine::OptixSimulationDataGeneric mem;
}

#endif // RMAGINE_SIMULATION_OPTIX_MEM_H