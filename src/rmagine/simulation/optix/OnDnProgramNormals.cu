#include <optix.h>
#include <math_constants.h>

#include "rmagine/math/math.h"
#include "rmagine/simulation/optix/OptixSimulationData.hpp"

using namespace rmagine;

extern "C" {
__constant__ OptixSimulationDataNormalsOnDn mem;
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
    
    const Transform Tsm = mem.Tbm[pid] * mem.Tsb[0];

    const Vector ray_orig_s = mem.model->getOrigin(vid, hid);
    const Vector ray_dir_s = mem.model->getDirection(vid, hid);

    const Vector ray_orig_m = Tsm * ray_orig_s;
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
    Tsm.R.x = __uint_as_float(optixGetPayload_1());
    Tsm.R.y = __uint_as_float(optixGetPayload_2());
    Tsm.R.z = __uint_as_float(optixGetPayload_3());
    Tsm.R.w = __uint_as_float(optixGetPayload_4());
    Tsm.t.x = __uint_as_float(optixGetPayload_5());
    Tsm.t.y = __uint_as_float(optixGetPayload_6());
    Tsm.t.z = __uint_as_float(optixGetPayload_7());
    const Transform Tms = Tsm.inv();

    // Get additional info
    const unsigned int face_id = optixGetPrimitiveIndex();
    const unsigned int object_id = optixGetInstanceIndex();
    
    const float3 dir_m = optixGetWorldRayDirection();
    const Vector ray_dir_m{dir_m.x, dir_m.y, dir_m.z};
    const Vector ray_dir_s = Tms.R * ray_dir_m;

    rmagine::HitGroupDataScene* hg_data  = reinterpret_cast<rmagine::HitGroupDataScene*>( optixGetSbtDataPointer() );
    
    const int mesh_id = hg_data->inst_to_mesh[object_id];
    const MeshAttributes* mesh_attr = &hg_data->mesh_attributes[mesh_id];

    const float3 normal = make_float3(
        mesh_attr->face_normals[face_id].x, 
        mesh_attr->face_normals[face_id].y, 
        mesh_attr->face_normals[face_id].z);
    const float3 normal_world = optixTransformNormalFromObjectToWorldSpace(normal);

    Vector nint{normal_world.x, normal_world.y, normal_world.z};
    nint.normalize();
    nint = Tms.R * nint;

    // flip?
    if(ray_dir_s.dot(nint) > 0.0)
    {
        nint *= -1.0;
    }

    mem.normals[glob_id] = nint.normalized();
}