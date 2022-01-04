#include <optix.h>
#include "imagine/math/math.h"
#include "imagine/simulation/optix/OptixSimulationData.hpp"

using namespace imagine;

extern "C" {
__constant__ OptixSimulationDataRangesOnDn mem;
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

    const unsigned int loc_id = mem.model->getBufferId(vid, hid);
    const unsigned int glob_id = pid * mem.model->size() + loc_id;
    

    // const Transform Tbm = mem.Tbm[pid];


    const Transform Tsm = mem.Tbm[pid] * mem.Tsb[0];
    

    const Vector ray_orig_s = mem.model->getOrigin(vid, hid);
    const Vector ray_dir_s = mem.model->getRay(vid, hid);


    const Vector ray_orig_m = Tsm * ray_orig_s;
    const Vector ray_dir_m = Tsm.R * ray_dir_s;

    // printf("vid %u, hid %u, model: %p, size: %u \n ", vid, hid, mem.model, mem.model->size());

    // printf("vid %u, hid %u, Tbm.R = %fx %fy %fz %fw , Tbm.t = %fx %fy %fz\n", vid, hid, Tbm.R.x, Tbm.R.y, Tbm.R.z, Tbm.R.w, Tbm.t.x, Tbm.t.y, Tbm.t.z);

    // printf("vid %u, hid %u -- orig: %f %f %f, dir: %f %f %f\n", vid, hid, ray_orig_m.x, ray_orig_m.y, ray_orig_m.z, ray_dir_m.x, ray_dir_m.y, ray_dir_m.z);

    unsigned int p0 = glob_id;
    optixTrace(
            mem.handle,
            make_float3(ray_orig_m.x, ray_orig_m.y, ray_orig_m.z ),
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