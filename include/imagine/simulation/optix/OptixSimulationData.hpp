#ifndef IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP
#define IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP

#include <imagine/util/optix/OptixData.hpp>
#include <imagine/types/sensor_models.h>
#include <imagine/types/types.h>

#include <optix.h>
#include <cuda_runtime.h>

namespace imagine
{

struct OptixSimulationDataRanges {
    // Input
    // Input
    const Transform*        Tsb; // Static offset of sensor
    const LiDARModel*       model; // Scanner Model
    const Transform*        Tbm; // Poses
    uint32_t                Nposes;
    OptixTraversableHandle  handle;
    // Result
    float*                  ranges;
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP