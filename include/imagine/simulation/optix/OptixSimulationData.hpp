#ifndef IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP
#define IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP

#include <imagine/util/optix/OptixData.hpp>
#include <imagine/types/sensor_models.h>
#include <imagine/math/types.h>

#include <optix.h>
#include <cuda_runtime.h>

namespace imagine
{

struct OptixSimulationDataRanges {
    // Input
    const Transform*        Tsb; // Static offset of sensor
    const LiDARModel*       model; // Scanner Model
    const Transform*        Tbm; // Poses
    uint32_t                Nposes;
    // Handle
    OptixTraversableHandle  handle;
    // Result
    float*                  ranges;
};

struct OptixSimulationDataNormals {
    // Input
    const Transform*        Tsb; // Static offset of sensor
    const LiDARModel*       model; // Scanner Model
    const Transform*        Tbm; // Poses
    uint32_t                Nposes;
    // Handle
    OptixTraversableHandle  handle;
    // Result
    Vector*                 normals;
};

struct OptixSimulationDataGeneric {
    // Input
    const Transform*        Tsb; // Static offset of sensor
    const LiDARModel*       model; // Scanner Model
    const Transform*        Tbm; // Poses
    uint32_t                Nposes;
    // Handle
    OptixTraversableHandle  handle;
    // Generic Options
    bool                    computeHits;
    bool                    computeRanges;
    bool                    computePoints;
    bool                    computeNormals;
    bool                    computeFaceIds;
    bool                    computeObjectIds;
    // Result
    uint8_t*                hits;
    float*                  ranges;
    Point*                  points;
    Vector*                 normals;
    unsigned int*           face_ids;
    unsigned int*           object_ids;
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP