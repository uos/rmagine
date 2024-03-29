#ifndef RMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP
#define RMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP

#include <rmagine/util/optix/OptixData.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>

#include <cuda_runtime.h>

namespace rmagine
{

template<typename ModelT>
struct OptixSimulationDataRanges {
    // Input
    const Transform*            Tsb; // Static offset of sensor
    const ModelT*               model; // Scanner Model
    const Transform*            Tbm; // Poses
    uint32_t                    Nposes;
    // Handle
    unsigned long long          handle;
    // Result
    float*                      ranges;
};

using OptixSimulationDataRangesSphere = OptixSimulationDataRanges<SphericalModel>;
using OptixSimulationDataRangesPinhole = OptixSimulationDataRanges<PinholeModel>;
using OptixSimulationDataRangesO1Dn = OptixSimulationDataRanges<O1DnModel_<VRAM_CUDA> >;
using OptixSimulationDataRangesOnDn = OptixSimulationDataRanges<OnDnModel_<VRAM_CUDA> >;

union SensorModelUnion
{
    SphericalModel* spherical;
    PinholeModel* pinhole;
    O1DnModel_<VRAM_CUDA>* o1dn;
    OnDnModel_<VRAM_CUDA>* ondn;
};

struct OptixSimulationDataGeneric {
    // Input
    const Transform*            Tsb; // Static offset of sensor
    uint32_t                    model_type; // Scanne model type id
    const SensorModelUnion*     model; // Scanner Model
    const Transform*            Tbm; // Poses
    uint32_t                    Nposes;
    // Handle
    unsigned long long      handle;
    // Generic Options
    bool                        computeHits;
    bool                        computeRanges;
    bool                        computePoints;
    bool                        computeNormals;
    bool                        computeFaceIds;
    bool                        computeGeomIds;
    bool                        computeObjectIds;
    // Result
    uint8_t*                    hits;
    float*                      ranges;
    Point*                      points;
    Vector*                     normals;
    unsigned int*               face_ids;
    unsigned int*               geom_ids;
    unsigned int*               object_ids;

    static OptixSimulationDataGeneric Zero()
    {
        OptixSimulationDataGeneric ret;
        ret.computeHits = false;
        ret.computeRanges = false;
        ret.computePoints = false;
        ret.computeNormals = false;
        ret.computeFaceIds = false;
        ret.computeGeomIds = false;
        ret.computeObjectIds = false;
        ret.hits = nullptr;
        ret.ranges = nullptr;
        ret.points = nullptr;
        ret.normals = nullptr;
        ret.face_ids = nullptr;
        ret.geom_ids = nullptr;
        ret.object_ids = nullptr;
        return ret;
    }
};

inline uint32_t get_bounding_id(const OptixSimulationDataGeneric& flags)
{
    uint32_t ret = 0;

    ret |= static_cast<uint32_t>(flags.computeHits) << 0;
    ret |= static_cast<uint32_t>(flags.computeRanges) << 1;
    ret |= static_cast<uint32_t>(flags.computePoints) << 2;
    ret |= static_cast<uint32_t>(flags.computeNormals) << 3;
    ret |= static_cast<uint32_t>(flags.computeFaceIds) << 4;
    ret |= static_cast<uint32_t>(flags.computeGeomIds) << 5;
    ret |= static_cast<uint32_t>(flags.computeObjectIds) << 6;

    return ret;
}

[[deprecated("Use get_bounding_id() instead.")]]
inline uint32_t boundingId(const OptixSimulationDataGeneric& flags)
{
    return get_bounding_id(flags);
}

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SIMULATION_DATA_HPP