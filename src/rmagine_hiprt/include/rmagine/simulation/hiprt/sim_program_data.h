/**
 * @file sim_program_data.h
 * @brief Simulation data structures for HIPRT ray-tracing backend
 *
 * Mirrors rmagine_optix/simulation/optix/sim_program_data.h
 * but without OptiX-specific types.
 */

#ifndef RMAGINE_SIMULATION_HIPRT_SIM_PROGRAM_DATA_H
#define RMAGINE_SIMULATION_HIPRT_SIM_PROGRAM_DATA_H

#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>

namespace rmagine
{

/**
 * @brief Device-side O1Dn model (one origin, N directions)
 *
 * The host O1DnModel_<RAM> contains Memory<Vector,RAM> dirs which cannot
 * be passed to the device. This struct holds device pointers.
 */
struct HiprtO1DnModelDevice
{
    uint32_t width;
    uint32_t height;
    float range_min;
    float range_max;
    Vector orig;        // single origin
    const Vector* dirs; // device pointer to directions array
};

/**
 * @brief Device-side OnDn model (N origins, N directions)
 *
 * The host OnDnModel_<RAM> contains Memory<Vector,RAM> for origs and dirs
 * which cannot be passed to the device. This struct holds device pointers.
 */
struct HiprtOnDnModelDevice
{
    uint32_t width;
    uint32_t height;
    float range_min;
    float range_max;
    const Vector* origs; // device pointer to origins array
    const Vector* dirs;  // device pointer to directions array
};

/**
 * @brief Union of sensor model pointers
 *
 * All sensor types (Pinhole, Spherical, O1Dn, OnDn) are supported.
 */
union HiprtSensorModelUnion
{
    PinholeModel* pinhole;
    SphericalModel* spherical;
    HiprtO1DnModelDevice* o1dn;
    HiprtOnDnModelDevice* ondn;
};

/**
 * @brief Per-mesh SBT-like data for HIPRT
 *
 * Unlike OptiX SBT records, these are passed to the trace kernel
 * as plain device pointers.
 */
struct HiprtMeshData
{
    Vector* vertex_normals = nullptr;
    Vector* face_normals = nullptr;
    unsigned int id = 0;
};

/**
 * @brief Scene geometry data for the trace kernel
 */
struct HiprtSceneData
{
    unsigned int n_geometries = 0;
    HiprtMeshData* geometries = nullptr;
};

/**
 * @brief Generic simulation data passed to HIPRT trace kernels
 *
 * This struct is passed as kernel arguments (not constant memory,
 * since HIPRT JIT kernels don't share a compile unit with us).
 */
struct HiprtSimulationData
{
    // Input: sensor pose
    const Transform* Tsb;           // Static sensor-to-body offset
    const Transform* Tbm;           // Body-to-map poses (Nposes elements)
    uint32_t Nposes;

    // Input: sensor model
    uint32_t model_type;            // Sensor type ID (0=Pinhole, 1=Sphere, etc.)
    const HiprtSensorModelUnion* model;

    // Input: geometry handle
    // For HIPRT this is hiprtGeometry (cast from unsigned long long)
    unsigned long long geom_handle;

    // Input: scene data for normals/IDs
    const HiprtSceneData* scene_data;

    // Output control flags
    bool computeHits;
    bool computeRanges;
    bool computePoints;
    bool computeNormals;
    bool computeFaceIds;
    bool computeGeomIds;
    bool computeObjectIds;

    // Output buffers (device pointers)
    uint8_t* hits;
    float* ranges;
    Point* points;
    Vector* normals;
    unsigned int* face_ids;
    unsigned int* geom_ids;
    unsigned int* object_ids;

    static HiprtSimulationData Zero()
    {
        HiprtSimulationData ret;
        ret.Tsb = nullptr;
        ret.Tbm = nullptr;
        ret.Nposes = 0;
        ret.model_type = 0;
        ret.model = nullptr;
        ret.geom_handle = 0;
        ret.scene_data = nullptr;
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

} // namespace rmagine

#endif // RMAGINE_SIMULATION_HIPRT_SIM_PROGRAM_DATA_H
