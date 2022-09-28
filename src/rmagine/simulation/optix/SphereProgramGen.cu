#include <optix.h>
#include "rmagine/math/types.h"
#include "rmagine/simulation/optix/sim_program_data.h"
#include "rmagine/map/optix/optix_sbt.h"

#include <math_constants.h>

#include "rmagine/simulation/optix/sim_program_mem.cuh"

using namespace rmagine;


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

    const SphericalModel* model = mem.model->spherical;
    const unsigned int loc_id = model->getBufferId(vid, hid);
    const unsigned int glob_id = pid * model->size() + loc_id;
    
    const Transform Tsm = mem.Tbm[pid] * mem.Tsb[0];

    const Vector ray_dir_s = model->getDirection(vid, hid);
    const Vector ray_dir_m = Tsm.R * ray_dir_s;

    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
    
    p0 = glob_id;
    p1 = __float_as_uint(Tsm.R.x);
    p2 = __float_as_uint(Tsm.R.y);
    p3 = __float_as_uint(Tsm.R.z);
    p4 = __float_as_uint(Tsm.R.w);
    p5 = __float_as_uint(Tsm.t.x);
    p6 = __float_as_uint(Tsm.t.y);
    p7 = __float_as_uint(Tsm.t.z);

    #if OPTIX_VERSION >= 70400
    optixTrace(
            OPTIX_PAYLOAD_TYPE_ID_0,
            mem.handle,
            make_float3(Tsm.t.x, Tsm.t.y, Tsm.t.z ),
            make_float3(ray_dir_m.x, ray_dir_m.y, ray_dir_m.z),
            0.0f,               // Min intersection distance
            model->range.max,                   // Max intersection distance
            0.0f,                       // rayTime -- used for motion blur
            OptixVisibilityMask( 1 ),   // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,          // SBT offset
            1,          // SBT stride
            0,          // missSBTIndex
            p0, p1, p2, p3, p4, p5, p6, p7 );
    #else
    optixTrace(
            mem.handle,
            make_float3(Tsm.t.x, Tsm.t.y, Tsm.t.z ),
            make_float3(ray_dir_m.x, ray_dir_m.y, ray_dir_m.z),
            0.0f,               // Min intersection distance
            model->range.max,                   // Max intersection distance
            0.0f,                       // rayTime -- used for motion blur
            OptixVisibilityMask( 1 ),   // Specify always visible
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,          // SBT offset
            1,          // SBT stride
            0,          // missSBTIndex
            p0, p1, p2, p3, p4, p5, p6, p7 );
    #endif
}
