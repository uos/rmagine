#include <optix.h>
// #include "rmagine/math/types.h"
// #include "rmagine/simulation/optix/sim_program_data.h"
// #include "rmagine/map/optix/optix_sbt.h"

#include <math_constants.h>

#include "rmagine/simulation/optix/sim_program_mem.cuh"

namespace rm = rmagine;


// TODO: Bug?
// 
// When compiling cuda in debug mode using nvcc flags -g -G, the following linker error occurs
// 
// OPTIX_ERROR_PIPELINE_LINK_ERROR
// Error: Symbol '_ZNK7rmagine11Quaternion_IfE3invEv' was defined multiple times. First seen in: '__raygen__rg'
// Error: Symbol '_ZNK7rmagine11Quaternion_IfE4multERKNS_8Vector3_IfEE' was defined multiple times. First seen in: '__raygen__rg'
// Error: Symbol '_ZNK7rmagine11Quaternion_IfE4multERKS1_' was defined multiple times. First seen in: '__raygen__rg'
// Error: Symbol '_ZNK7rmagine11Quaternion_IfEmlERKNS_8Vector3_IfEE' was defined multiple times. First seen in: '__raygen__rg'
// 
// The following commented lines are for debugging purposes. With them the linker error vanishes

// namespace sphere_program_gen {

// __device__
// rm::Quaternion mult(const rm::Quaternion& q1, const rm::Quaternion& q2)
// {
//     rm::Quaternion q3;
//     q3.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
//     q3.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
//     q3.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
//     q3.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
//     return q3;
// }

// __device__
// rm::Vector mult(const rm::Quaternion& q, const rm::Vector& p)
// {
//     const rm::Quaternion P{p.x, p.y, p.z, 0.0};
//     const rm::Quaternion PT = mult(q, P);
//     return {PT.x, PT.y, PT.z};
// }


// __device__
// rm::Transform mult(const rm::Transform& T1, const rm::Transform& T2)
// {
//     rm::Transform T3;
//     T3.t = mult(T1.R, T2.t);
//     T3.R = mult(T1.R, T2.R);
//     T3.t += T1.t;
//     return T3;
// }

// } // namespace sphere_program_gen

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

    const rm::SphericalModel* model = mem.model->spherical;
    const unsigned int loc_id = model->getBufferId(vid, hid);
    const unsigned int glob_id = pid * model->size() + loc_id;
    
    const rm::Transform Tsm = mem.Tbm[pid] * mem.Tsb[0];
    // const rm::Transform Tsm = sphere_program_gen::mult(mem.Tbm[pid], mem.Tsb[0]);

    const rm::Vector ray_dir_s = model->getDirection(vid, hid);
    const rm::Vector ray_dir_m = Tsm.R * ray_dir_s;
    // const rm::Vector ray_dir_m = sphere_program_gen::mult(Tsm.R, ray_dir_s);

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
