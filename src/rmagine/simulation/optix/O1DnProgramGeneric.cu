#include <optix.h>
#include "rmagine/math/types.h"
#include "rmagine/simulation/optix/OptixSimulationData.hpp"
#include "rmagine/map/optix/optix_sbt.h"

#include <math_constants.h>

using namespace rmagine;

extern "C" {
__constant__ OptixSimulationDataGenericO1Dn mem;
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



__forceinline__ __device__
void computeHit()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.hits[glob_id] = 1;
}

__forceinline__ __device__
void computeNoHit()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.hits[glob_id] = 0;
}

__forceinline__ __device__
void computeRange()
{
    const unsigned int glob_id = optixGetPayload_0();
    const float t = optixGetRayTmax();
    mem.ranges[glob_id] = t;
}

__forceinline__ __device__
void computeNoRange()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.ranges[glob_id] = mem.model->range.max + 1.0f;
}

__forceinline__ __device__
void computePoint()
{
    // compute points in sensor space
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

    const float3 pos_m = optixGetWorldRayOrigin();
    const float3 dir_m = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();

    const Vector ray_pos_m{pos_m.x, pos_m.y, pos_m.z};
    const Vector ray_dir_m{dir_m.x, dir_m.y, dir_m.z};

    const Vector ray_pos_s = Tms * ray_pos_m;
    const Vector ray_dir_s = Tms.R * ray_dir_m;

    mem.points[glob_id] = ray_pos_s + ray_dir_s * t;
}

__forceinline__ __device__
void computeNoPoint()
{
    const unsigned int glob_id = optixGetPayload_0();

    mem.points[glob_id] = {
        CUDART_NAN_F,
        CUDART_NAN_F,
        CUDART_NAN_F
    };
}

__forceinline__ __device__
void computeNormalSBT()
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
    const unsigned int inst_id = optixGetInstanceId();
    const unsigned int gas_id = optixGetSbtGASIndex();
    

    const float3 dir_m = optixGetWorldRayDirection();
    const Vector ray_dir_m{dir_m.x, dir_m.y, dir_m.z};
    const Vector ray_dir_s = Tms.R * ray_dir_m;

    OptixSceneSBT* scene_data  = reinterpret_cast<OptixSceneSBT*>( optixGetSbtDataPointer() );

    OptixMeshSBT* mesh_data = nullptr;
    if(scene_data->type == OptixSceneType::INSTANCES)
    {
        // instance hierarchy
        OptixSceneSBT* inst_scene = scene_data->geometries[inst_id].inst_data.scene;
        mesh_data = &(inst_scene->geometries[gas_id].mesh_data);
    } else {
        mesh_data = &scene_data->geometries[gas_id].mesh_data;
    }

    const float3 normal = make_float3(
        mesh_data->face_normals[face_id].x, 
        mesh_data->face_normals[face_id].y, 
        mesh_data->face_normals[face_id].z);
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

__forceinline__ __device__
void computeNormal()
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

    const float3 dir_m = optixGetWorldRayDirection();
    const Vector ray_dir_m{dir_m.x, dir_m.y, dir_m.z};
    const Vector ray_dir_s = Tms.R * ray_dir_m;


    // Get additional info
    const unsigned int face_id = optixGetPrimitiveIndex();
    const unsigned int object_id = optixGetInstanceId();
    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gas_id = optixGetSbtGASIndex();
    float time = optixGetRayTime();

    // fetch vertices of face
    float3 vertices[3];
    optixGetTriangleVertexData(gas, face_id, gas_id, time, vertices);

    // compute normal
    const Vector v0 = {vertices[0].x, vertices[0].y, vertices[0].z};
    const Vector v1 = {vertices[1].x, vertices[1].y, vertices[1].z};
    const Vector v2 = {vertices[2].x, vertices[2].y, vertices[2].z};

    const Vector3 rm_normal = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();

    const float3 normal = make_float3(rm_normal.x, rm_normal.y, rm_normal.z);
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

__forceinline__ __device__
void computeNoNormal()
{
    const unsigned int glob_id = optixGetPayload_0();

    mem.normals[glob_id] = {
        CUDART_NAN_F,
        CUDART_NAN_F,
        CUDART_NAN_F
    };
}

__forceinline__ __device__
void computeFaceId()
{
    const unsigned int glob_id = optixGetPayload_0();
    const unsigned int face_id = optixGetPrimitiveIndex();
    mem.face_ids[glob_id] = face_id;
}

__forceinline__ __device__
void computeNoFaceId()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.face_ids[glob_id] = __UINT_MAX__;
}

__forceinline__ __device__
void computeGeomId()
{
    const unsigned int glob_id = optixGetPayload_0();

    const unsigned int inst_id = optixGetInstanceId();
    const unsigned int sbt_gas_id = optixGetSbtGASIndex();

    unsigned int geom_id = 0;
    // printf("Inst %u, SBT GAS %u \n", inst_id, sbt_gas_id);

    OptixSceneSBT* scene_data  = reinterpret_cast<OptixSceneSBT*>( optixGetSbtDataPointer() );
    if(scene_data->type == OptixSceneType::INSTANCES)
    {
        // instance hierarchy
        geom_id = scene_data->geometries[inst_id].inst_data.scene->geometries[sbt_gas_id].mesh_data.id;
    } else {
        geom_id = scene_data->geometries[sbt_gas_id].mesh_data.id;
    }

    mem.geom_ids[glob_id] = geom_id;
}

__forceinline__ __device__
void computeNoGeomId()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.geom_ids[glob_id] = __UINT_MAX__;
}

__forceinline__ __device__
void computeObjectId()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.object_ids[glob_id] = optixGetInstanceId();
}

__forceinline__ __device__
void computeNoObjectId()
{
    const unsigned int glob_id = optixGetPayload_0();
    mem.object_ids[glob_id] = __UINT_MAX__;
}

extern "C" __global__ void __miss__ms()
{
    if(mem.computeHits)
    {
        computeNoHit();
    }

    if(mem.computeRanges)
    {
        computeNoRange();
    }

    if(mem.computePoints)
    {
        computeNoPoint();
    }

    if(mem.computeNormals)
    {
        computeNoNormal();
    }

    if(mem.computeFaceIds)
    {
        computeNoFaceId();
    }

    if(mem.computeGeomIds)
    {
        computeNoGeomId();
    }

    if(mem.computeObjectIds)
    {
        computeNoObjectId();
    }
}

extern "C" __global__ void __closesthit__ch()
{
    if(mem.computeHits)
    {
        computeHit();
    }

    if(mem.computeRanges)
    {
        computeRange();
    }

    if(mem.computePoints)
    {
        computePoint();
    }

    if(mem.computeNormals)
    {
        computeNormalSBT();
    }

    if(mem.computeFaceIds)
    {
        computeFaceId();
    }

    if(mem.computeGeomIds)
    {
        computeGeomId();
    }

    if(mem.computeObjectIds)
    {
        computeObjectId();
    }
}