#include <optix.h>
#include "imagine/math/math.h"
#include "imagine/simulation/optix/OptixSimulationData.hpp"

using namespace imagine;

extern "C" {
__constant__ OptixSimulationDataNormals mem;
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

    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
    
    p0 = glob_id;
    p1 = float_as_uint(Tsm.R.x);
    p2 = float_as_uint(Tsm.R.y);
    p3 = float_as_uint(Tsm.R.z);
    p4 = float_as_uint(Tsm.R.w);
    p5 = float_as_uint(Tsm.t.x);
    p6 = float_as_uint(Tsm.t.y);
    p7 = float_as_uint(Tsm.t.z);

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
            1,          // SBT stride
            0,          // missSBTIndex
            p0, p1, p2, p3, p4, p5, p6, p7 );
}

extern "C" __global__ void __miss__ms()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.normals[glob_id] = {
        mem.model->range.max + 1.0f,
        mem.model->range.max + 1.0f,
        mem.model->range.max + 1.0f
    };
}

extern "C" __global__ void __closesthit__ch()
{
    // Get Payloads
    const unsigned int glob_id = optixGetPayload_0();
    Transform Tsm;
    Tsm.R.x = uint_as_float(optixGetPayload_1());
    Tsm.R.y = uint_as_float(optixGetPayload_2());
    Tsm.R.z = uint_as_float(optixGetPayload_3());
    Tsm.R.w = uint_as_float(optixGetPayload_4());
    Tsm.t.x = uint_as_float(optixGetPayload_5());
    Tsm.t.y = uint_as_float(optixGetPayload_6());
    Tsm.t.z = uint_as_float(optixGetPayload_7());
    const Transform Tms = Tsm.inv();

    // Get additionals info
    const unsigned int face_id = optixGetPrimitiveIndex();
    const unsigned int object_id = optixGetInstanceIndex();
    const float3 dir_m = optixGetWorldRayDirection();
    const Vector ray_dir_m{dir_m.x, dir_m.y, dir_m.z};

    const Vector ray_dir_s = Tms.R * ray_dir_m;
    

    // const Vector ray_dir_m =

    // Test to receive the normal as attribute instead. like barycentrics  
    // doesnt work: attributes are fixed for primitive types. in this case triangles: only barycentrics  
    imagine::HitGroupDataNormals* hg_data  = reinterpret_cast<imagine::HitGroupDataNormals*>( optixGetSbtDataPointer() );

    float3 normal = make_float3(hg_data->normals[object_id][face_id].x, hg_data->normals[object_id][face_id].y, hg_data->normals[object_id][face_id].z);
    
    float3 normal_world = optixTransformNormalFromObjectToWorldSpace(normal);

    

    Vector nint{normal_world.x, normal_world.y, normal_world.z};
    nint.normalize();
    nint = Tms.R * nint;

    if(ray_dir_s.dot(nint) > 0.0)
    {
        nint *= -1.0;
    }

    mem.normals[glob_id] = nint.normalized();
}