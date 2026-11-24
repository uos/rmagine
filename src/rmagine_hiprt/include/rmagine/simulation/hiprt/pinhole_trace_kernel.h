/**
 * @file pinhole_trace_kernel.h
 * @brief HIPRT trace kernel for Pinhole sensor
 *
 * This file is JIT-compiled by hiprtc at runtime. It contains
 * the trace kernel that replaces OptiX's raygen/closesthit/miss
 * program model with a single HIP kernel.
 *
 * The kernel is called by hiprtBuildTraceKernels and launched
 * via oroModuleLaunchKernel.
 */

#ifndef RMAGINE_HIPRT_PINHOLE_TRACE_KERNEL_H
#define RMAGINE_HIPRT_PINHOLE_TRACE_KERNEL_H

#include <hiprt/hiprt_device.h>

// Math types (must match rmagine's types.h)
struct Vector3f
{
    float x, y, z;
};

struct Quaternionf
{
    float x, y, z, w;
};

struct Transform3f
{
    Quaternionf R;          // rotation as quaternion
    Vector3f t;             // translation
    unsigned int stamp;     // rmagine::Transform_ has this field
};

struct PinholeModelDev
{
    unsigned int width;
    unsigned int height;
    float f_x, f_y;         // focal lengths
    float c_x, c_y;         // principal point
    float range_min, range_max;
};

// Device math helpers
__device__ __forceinline__
Vector3f quat_rotate(const Quaternionf& q, const Vector3f& v)
{
    // Rodrigues rotation: v' = v + 2*w*(qv x v) + 2*(qv x (qv x v))
    Vector3f qv = {q.x, q.y, q.z};

    // t = 2 * cross(qv, v)
    Vector3f t;
    t.x = 2.0f * (qv.y * v.z - qv.z * v.y);
    t.y = 2.0f * (qv.z * v.x - qv.x * v.z);
    t.z = 2.0f * (qv.x * v.y - qv.y * v.x);

    // result = v + w*t + cross(qv, t)
    Vector3f result;
    result.x = v.x + q.w * t.x + (qv.y * t.z - qv.z * t.y);
    result.y = v.y + q.w * t.y + (qv.z * t.x - qv.x * t.z);
    result.z = v.z + q.w * t.z + (qv.x * t.y - qv.y * t.x);

    return result;
}

__device__ __forceinline__
Vector3f transform_point(const Transform3f& T, const Vector3f& p)
{
    Vector3f rotated = quat_rotate(T.R, p);
    return {rotated.x + T.t.x, rotated.y + T.t.y, rotated.z + T.t.z};
}

__device__ __forceinline__
Vector3f transform_vector(const Transform3f& T, const Vector3f& v)
{
    return quat_rotate(T.R, v);
}

__device__ __forceinline__
float dot(const Vector3f& a, const Vector3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__
Vector3f normalize(const Vector3f& v)
{
    float len = sqrtf(dot(v, v));
    if (len > 0.0f) {
        float inv = 1.0f / len;
        return {v.x * inv, v.y * inv, v.z * inv};
    }
    return v;
}

// Simulation parameters (passed as kernel args)
struct PinholeTraceParams
{
    // Input
    const Transform3f* Tsb;         // sensor-to-body (1 element)
    const Transform3f* Tbm;         // body-to-map (Nposes elements)
    unsigned int Nposes;

    const PinholeModelDev* model;

    // Geometry handle (cast to hiprtGeometry)
    hiprtGeometry geom;

    // Mesh data for normals
    const Vector3f* face_normals;   // can be null
    unsigned int n_meshes;

    // Output control
    bool computeHits;
    bool computeRanges;
    bool computePoints;
    bool computeNormals;

    // Output buffers
    unsigned char* hits;
    float* ranges;
    Vector3f* points;
    Vector3f* normals;
};

/**
 * @brief Pinhole trace kernel
 *
 * Launch dimensions: (width, height, Nposes)
 * Each thread traces one ray and writes results.
 */
extern "C" __global__ void pinhole_trace_kernel(PinholeTraceParams params)
{
    // Thread indices
    const unsigned int hid = blockIdx.x * blockDim.x + threadIdx.x;  // horizontal (column)
    const unsigned int vid = blockIdx.y * blockDim.y + threadIdx.y;  // vertical (row)
    const unsigned int pid = blockIdx.z;                              // pose index

    const PinholeModelDev* model = params.model;

    // Bounds check
    if (hid >= model->width || vid >= model->height || pid >= params.Nposes) {
        return;
    }

    // Compute ray index
    const unsigned int loc_id = vid * model->width + hid;
    const unsigned int glob_id = pid * (model->width * model->height) + loc_id;

    // Compute ray direction in sensor frame
    // Pinhole model: ray through pixel (hid, vid)
    Vector3f ray_dir_s;
    ray_dir_s.x = (float(hid) - model->c_x) / model->f_x;
    ray_dir_s.y = (float(vid) - model->c_y) / model->f_y;
    ray_dir_s.z = 1.0f;
    ray_dir_s = normalize(ray_dir_s);

    // Compose transforms: Tsm = Tbm * Tsb
    Transform3f Tsm;
    {
        Transform3f Tsb = params.Tsb[0];
        Transform3f Tbm = params.Tbm[pid];

        // Rotation: Rsm = Rbm * Rsb
        // Quaternion multiplication: q = q1 * q2
        Quaternionf q1 = Tbm.R;
        Quaternionf q2 = Tsb.R;
        Tsm.R.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
        Tsm.R.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
        Tsm.R.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
        Tsm.R.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

        // Translation: tsm = Rbm * tsb + tbm
        Vector3f tsb_rotated = quat_rotate(Tbm.R, Tsb.t);
        Tsm.t.x = tsb_rotated.x + Tbm.t.x;
        Tsm.t.y = tsb_rotated.y + Tbm.t.y;
        Tsm.t.z = tsb_rotated.z + Tbm.t.z;
    }

    // Transform ray to map frame
    Vector3f ray_origin = Tsm.t;
    Vector3f ray_dir_m = transform_vector(Tsm, ray_dir_s);

    // Create HIPRT ray
    hiprtRay ray;
    ray.origin = make_float3(ray_origin.x, ray_origin.y, ray_origin.z);
    ray.direction = make_float3(ray_dir_m.x, ray_dir_m.y, ray_dir_m.z);
    ray.minT = model->range_min;
    ray.maxT = model->range_max;

    // Trace ray using HIPRT
    hiprtGeomTraversalClosest tr(params.geom, ray);
    hiprtHit hit = tr.getNextHit();

    // Process hit/miss
    if (hit.hasHit())
    {
        // HIT
        float t = hit.t;
        unsigned int primID = hit.primID;

        if (params.computeHits && params.hits) {
            params.hits[glob_id] = 1;
        }

        if (params.computeRanges && params.ranges) {
            params.ranges[glob_id] = t;
        }

        if (params.computePoints && params.points) {
            // Compute hit point in sensor frame
            Vector3f hit_m;
            hit_m.x = ray_origin.x + ray_dir_m.x * t;
            hit_m.y = ray_origin.y + ray_dir_m.y * t;
            hit_m.z = ray_origin.z + ray_dir_m.z * t;

            // Transform to sensor frame (Tms = inv(Tsm))
            // For points: p_s = R_sm^T * (p_m - t_sm)
            Vector3f diff;
            diff.x = hit_m.x - Tsm.t.x;
            diff.y = hit_m.y - Tsm.t.y;
            diff.z = hit_m.z - Tsm.t.z;

            // Inverse rotation: R^T * v = conjugate(q) * v
            Quaternionf q_inv = {-Tsm.R.x, -Tsm.R.y, -Tsm.R.z, Tsm.R.w};
            Vector3f hit_s = quat_rotate(q_inv, diff);

            params.points[glob_id] = hit_s;
        }

        if (params.computeNormals && params.normals && params.face_normals) {
            // Get face normal and transform to sensor frame
            Vector3f normal_m = params.face_normals[primID];

            // Transform to sensor frame
            Quaternionf q_inv = {-Tsm.R.x, -Tsm.R.y, -Tsm.R.z, Tsm.R.w};
            Vector3f normal_s = quat_rotate(q_inv, normal_m);

            // Flip if facing away
            if (dot(ray_dir_s, normal_s) > 0.0f) {
                normal_s.x = -normal_s.x;
                normal_s.y = -normal_s.y;
                normal_s.z = -normal_s.z;
            }

            params.normals[glob_id] = normalize(normal_s);
        }
    }
    else
    {
        // MISS
        if (params.computeHits && params.hits) {
            params.hits[glob_id] = 0;
        }

        if (params.computeRanges && params.ranges) {
            params.ranges[glob_id] = model->range_max + 1.0f;
        }

        if (params.computePoints && params.points) {
            float nan = __int_as_float(0x7fc00000);  // quiet NaN
            params.points[glob_id] = {nan, nan, nan};
        }

        if (params.computeNormals && params.normals) {
            float nan = __int_as_float(0x7fc00000);
            params.normals[glob_id] = {nan, nan, nan};
        }
    }
}

#endif // RMAGINE_HIPRT_PINHOLE_TRACE_KERNEL_H
