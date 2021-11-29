#include <optix.h>
#include "imagine/math/math.h"
#include "imagine/simulation/optix/OptixSimulationData.hpp"

using namespace imagine;

extern "C" {
__constant__ OptixSimulationDataRanges mem;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // vertical id
    const unsigned int hid = idx.x;
    // horizontal id
    const unsigned int vid = idx.y;
    // pose id
    const unsigned int pid = idx.z;

    const unsigned int loc_id = vid * mem.model->theta.size + hid;
    const unsigned int glob_id = pid * mem.model->theta.size * mem.model->phi.size + loc_id;
    
    const Transform Tsm = mem.Tbm[pid] * mem.Tsb[0];

    const Vector ray_dir_s = mem.model->getRay(vid, hid);
    const Vector ray_dir_m = Tsm.R * ray_dir_s;

    unsigned int p0 = glob_id;
    optixTrace(
            mem.handle,
            make_float3(Tsm.t.x, Tsm.t.y, Tsm.t.z ),
            make_float3(ray_dir_m.x, ray_dir_m.y, ray_dir_m.z),
            0.0f,               // Min intersection distance
            mem.model->range.max,                   // Max intersection distance
            0.0f,                       // rayTime -- used for motion blur
            OptixVisibilityMask( 1 ),   // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,          // SBT offset
            1,             // SBT stride
            0,          // missSBTIndex 
            p0);
}

extern "C" __global__ void __miss__ms()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.ranges[glob_id] = mem.model->range.max + 1.0f;
}

extern "C" __global__ void __closesthit__ch()
{
    const float t = optixGetRayTmax();
    const unsigned int glob_id = optixGetPayload_0();
    mem.ranges[glob_id] = t;
}