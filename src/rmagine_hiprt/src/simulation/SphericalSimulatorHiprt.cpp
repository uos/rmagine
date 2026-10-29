/**
 * @file SphericalSimulatorHiprt.cpp
 * @brief Spherical (lidar) simulator implementation using HIPRT
 *
 * JIT-compiles a trace kernel using hiprtBuildTraceKernels and launches
 * it via the HIP driver API (hipModuleLaunchKernel).
 *
 * Ray directions are computed from spherical coordinates:
 *   dir.x = cos(phi) * cos(theta)
 *   dir.y = cos(phi) * sin(theta)
 *   dir.z = sin(phi)
 * where phi (vertical/pitch) and theta (horizontal/yaw) are discretized
 * according to the SphericalModel.
 */

#include "rmagine/simulation/SphericalSimulatorHiprt.hpp"

#include <hiprt/hiprt.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace rmagine
{

SphericalSimulatorHiprt::SphericalSimulatorHiprt()
    : m_Tsb(1)
{
    Memory<Transform, RAM> I(1);
    I[0].setIdentity();
    m_Tsb = I;
}

SphericalSimulatorHiprt::SphericalSimulatorHiprt(HiprtScenePtr scene)
    : SphericalSimulatorHiprt()
{
    setScene(scene);
}

SphericalSimulatorHiprt::~SphericalSimulatorHiprt()
{
    m_kernel_module = nullptr;
    m_kernel_func = nullptr;
}

void SphericalSimulatorHiprt::setScene(HiprtScenePtr scene)
{
    m_scene = scene;
    m_kernel_func = nullptr;
}

void SphericalSimulatorHiprt::setModel(const SphericalModel& model)
{
    Memory<SphericalModel, RAM> tmp(1);
    tmp[0] = model;
    setModel(tmp);
}

void SphericalSimulatorHiprt::setModel(const Memory<SphericalModel, RAM>& model)
{
    m_width = model[0].getWidth();
    m_height = model[0].getHeight();
    m_model = model;

    Memory<HiprtSensorModelUnion, RAM> model_union(1);
    model_union[0].spherical = m_model.raw();
    m_model_union = model_union;
}

void SphericalSimulatorHiprt::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}

void SphericalSimulatorHiprt::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

Memory<float, VRAM_CUDA> SphericalSimulatorHiprt::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    const size_t nposes = Tbm.size();
    const size_t nrays = m_width * m_height * nposes;

    Memory<float, VRAM_CUDA> ranges(nrays);

    HiprtSimulationData data = HiprtSimulationData::Zero();
    data.computeRanges = true;
    data.ranges = ranges.raw();

    simulate(Tbm, data);

    return ranges;
}

Memory<uint8_t, VRAM_CUDA> SphericalSimulatorHiprt::simulateHits(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    const size_t nposes = Tbm.size();
    const size_t nrays = m_width * m_height * nposes;

    Memory<uint8_t, VRAM_CUDA> hits(nrays);

    HiprtSimulationData data = HiprtSimulationData::Zero();
    data.computeHits = true;
    data.hits = hits.raw();

    simulate(Tbm, data);

    return hits;
}

Memory<Point, VRAM_CUDA> SphericalSimulatorHiprt::simulatePoints(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    const size_t nposes = Tbm.size();
    const size_t nrays = m_width * m_height * nposes;

    Memory<Point, VRAM_CUDA> points(nrays);

    HiprtSimulationData data = HiprtSimulationData::Zero();
    data.computePoints = true;
    data.points = points.raw();

    simulate(Tbm, data);

    return points;
}

void SphericalSimulatorHiprt::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    HiprtSimulationData& data)
{
    if (!m_scene) {
        throw std::runtime_error("SphericalSimulatorHiprt: no scene set");
    }

    ensureKernel();

    data.Tsb = m_Tsb.raw();
    data.Tbm = Tbm.raw();
    data.Nposes = Tbm.size();
    data.model_type = 1;  // Spherical
    data.model = m_model_union.raw();
    data.geom_handle = m_scene->handle();
    data.scene_data = &m_scene->sceneData();

    launchKernel(data, m_width, m_height, Tbm.size());
}

void SphericalSimulatorHiprt::ensureKernel()
{
    if (m_kernel_func) {
        return;
    }

    const char* hiprt_path_env = std::getenv("HIPRT_PATH");
    if (!hiprt_path_env) {
        throw std::runtime_error(
            "HIPRT_PATH environment variable not set. "
            "Set it to the HIPRT SDK root (e.g., /path/to/HIPRT).");
    }
    std::string hiprt_path = hiprt_path_env;

    // Spherical model kernel source
    std::string kernel_src = R"(
#include <hiprt/hiprt_device.h>

struct Vector3f { float x, y, z; };
struct Quaternionf { float x, y, z, w; };
struct Transform3f {
    Quaternionf R;
    Vector3f t;
    unsigned int stamp;
};

// SphericalModel layout matches rmagine::SphericalModel
// phi: vertical (pitch), theta: horizontal (yaw)
struct DiscreteIntervalDev {
    float min;
    float inc;
    unsigned int size;
};

struct IntervalDev {
    float min;
    float max;
};

struct SphericalModelDev {
    DiscreteIntervalDev phi;    // vertical
    DiscreteIntervalDev theta;  // horizontal
    IntervalDev range;
};

__device__ __forceinline__ Vector3f quat_rotate(const Quaternionf& q, const Vector3f& v) {
    Vector3f qv = {q.x, q.y, q.z};
    Vector3f t;
    t.x = 2.0f * (qv.y * v.z - qv.z * v.y);
    t.y = 2.0f * (qv.z * v.x - qv.x * v.z);
    t.z = 2.0f * (qv.x * v.y - qv.y * v.x);
    Vector3f result;
    result.x = v.x + q.w * t.x + (qv.y * t.z - qv.z * t.y);
    result.y = v.y + q.w * t.y + (qv.z * t.x - qv.x * t.z);
    result.z = v.z + q.w * t.z + (qv.x * t.y - qv.y * t.x);
    return result;
}

__device__ __forceinline__ float dot3(const Vector3f& a, const Vector3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vector3f normalize3(const Vector3f& v) {
    float len = sqrtf(dot3(v, v));
    if (len > 0.0f) {
        float inv = 1.0f / len;
        return {v.x * inv, v.y * inv, v.z * inv};
    }
    return v;
}

struct SphericalTraceParams {
    const Transform3f* Tsb;
    const Transform3f* Tbm;
    unsigned int Nposes;
    const SphericalModelDev* model;
    hiprtGeometry geom;
    const Vector3f* face_normals;
    unsigned int n_meshes;
    bool computeHits;
    bool computeRanges;
    bool computePoints;
    bool computeNormals;
    unsigned char* hits;
    float* ranges;
    Vector3f* points;
    Vector3f* normals;
};

extern "C" __global__ void spherical_trace_kernel(SphericalTraceParams params)
{
    // theta_id = horizontal (width), phi_id = vertical (height)
    const unsigned int theta_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int phi_id = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pid = blockIdx.z;

    const SphericalModelDev* model = params.model;
    const unsigned int width = model->theta.size;
    const unsigned int height = model->phi.size;

    if (theta_id >= width || phi_id >= height || pid >= params.Nposes) return;

    const unsigned int loc_id = phi_id * width + theta_id;
    const unsigned int glob_id = pid * (width * height) + loc_id;

    // Compute spherical coordinates
    float phi = model->phi.min + float(phi_id) * model->phi.inc;
    float theta = model->theta.min + float(theta_id) * model->theta.inc;

    // Ray direction in sensor frame (spherical to Cartesian)
    Vector3f ray_dir_s;
    ray_dir_s.x = cosf(phi) * cosf(theta);
    ray_dir_s.y = cosf(phi) * sinf(theta);
    ray_dir_s.z = sinf(phi);

    // Compose Tsm = Tbm * Tsb
    Transform3f Tsb = params.Tsb[0];
    Transform3f Tbm = params.Tbm[pid];
    Transform3f Tsm;
    Quaternionf q1 = Tbm.R, q2 = Tsb.R;
    Tsm.R.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    Tsm.R.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    Tsm.R.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    Tsm.R.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    Vector3f tsb_rot = quat_rotate(Tbm.R, Tsb.t);
    Tsm.t.x = tsb_rot.x + Tbm.t.x;
    Tsm.t.y = tsb_rot.y + Tbm.t.y;
    Tsm.t.z = tsb_rot.z + Tbm.t.z;

    Vector3f ray_origin = Tsm.t;
    Vector3f ray_dir_m = quat_rotate(Tsm.R, ray_dir_s);

    hiprtRay ray;
    ray.origin = make_float3(ray_origin.x, ray_origin.y, ray_origin.z);
    ray.direction = make_float3(ray_dir_m.x, ray_dir_m.y, ray_dir_m.z);
    ray.minT = model->range.min;
    ray.maxT = model->range.max;

    hiprtGeomTraversalClosest tr(params.geom, ray);
    hiprtHit hit = tr.getNextHit();

    if (hit.hasHit()) {
        float t = hit.t;
        if (params.computeHits && params.hits) params.hits[glob_id] = 1;
        if (params.computeRanges && params.ranges) params.ranges[glob_id] = t;

        if (params.computePoints && params.points) {
            Vector3f hit_m = {ray_origin.x + ray_dir_m.x * t,
                              ray_origin.y + ray_dir_m.y * t,
                              ray_origin.z + ray_dir_m.z * t};
            Vector3f diff = {hit_m.x - Tsm.t.x, hit_m.y - Tsm.t.y, hit_m.z - Tsm.t.z};
            Quaternionf q_inv = {-Tsm.R.x, -Tsm.R.y, -Tsm.R.z, Tsm.R.w};
            params.points[glob_id] = quat_rotate(q_inv, diff);
        }
        if (params.computeNormals && params.normals && params.face_normals) {
            Vector3f normal_m = params.face_normals[hit.primID];
            Quaternionf q_inv = {-Tsm.R.x, -Tsm.R.y, -Tsm.R.z, Tsm.R.w};
            Vector3f normal_s = quat_rotate(q_inv, normal_m);
            if (dot3(ray_dir_s, normal_s) > 0.0f) {
                normal_s.x = -normal_s.x; normal_s.y = -normal_s.y; normal_s.z = -normal_s.z;
            }
            params.normals[glob_id] = normalize3(normal_s);
        }
    } else {
        if (params.computeHits && params.hits) params.hits[glob_id] = 0;
        if (params.computeRanges && params.ranges) params.ranges[glob_id] = model->range.max + 1.0f;
        if (params.computePoints && params.points) {
            float nan = __int_as_float(0x7fc00000);
            params.points[glob_id] = {nan, nan, nan};
        }
        if (params.computeNormals && params.normals) {
            float nan = __int_as_float(0x7fc00000);
            params.normals[glob_id] = {nan, nan, nan};
        }
    }
}
)";

    hiprtContext ctx = m_scene->context()->handle();

    const char* func_names[] = {"spherical_trace_kernel"};
    hiprtApiFunction funcs[1];
    hiprtApiModule module = nullptr;

    std::string hiprt_device_header = hiprt_path + "/hiprt";

    std::vector<std::string> options_str = {
        "-I" + hiprt_device_header,
        "-I" + hiprt_path,
        "--gpu-max-threads-per-block=256"
    };
    std::vector<const char*> options;
    for (const auto& opt : options_str) {
        options.push_back(opt.c_str());
    }

    hiprtError err = hiprtBuildTraceKernels(
        ctx,
        1,
        func_names,
        kernel_src.c_str(),
        "spherical_trace",
        0, nullptr, nullptr,
        (uint32_t)options.size(),
        options.data(),
        0, 0, nullptr,
        funcs,
        &module,
        true
    );

    if (err != hiprtSuccess) {
        throw std::runtime_error(
            "hiprtBuildTraceKernels failed with error " + std::to_string(err));
    }

    m_kernel_module = module;
    m_kernel_func = reinterpret_cast<void*>(funcs[0]);

    std::cout << "[SphericalSimulatorHiprt] JIT kernel compiled successfully" << std::endl;
}

void SphericalSimulatorHiprt::launchKernel(
    const HiprtSimulationData& data,
    unsigned int width,
    unsigned int height,
    unsigned int nposes)
{
    if (!m_kernel_func) {
        throw std::runtime_error("Kernel not compiled");
    }

    struct KernelParams {
        const void* Tsb;
        const void* Tbm;
        unsigned int Nposes;
        const void* model;
        void* geom;
        const void* face_normals;
        unsigned int n_meshes;
        bool computeHits;
        bool computeRanges;
        bool computePoints;
        bool computeNormals;
        unsigned char* hits;
        float* ranges;
        void* points;
        void* normals;
    };

    KernelParams params;
    params.Tsb = data.Tsb;
    params.Tbm = data.Tbm;
    params.Nposes = data.Nposes;
    params.model = data.model->spherical;
    params.geom = reinterpret_cast<void*>(data.geom_handle);
    params.face_normals = data.scene_data ? data.scene_data->geometries[0].face_normals : nullptr;
    params.n_meshes = data.scene_data ? data.scene_data->n_geometries : 0;
    params.computeHits = data.computeHits;
    params.computeRanges = data.computeRanges;
    params.computePoints = data.computePoints;
    params.computeNormals = data.computeNormals;
    params.hits = data.hits;
    params.ranges = data.ranges;
    params.points = data.points;
    params.normals = data.normals;

    const unsigned int blockX = 16, blockY = 16;
    unsigned int gridX = (width + blockX - 1) / blockX;
    unsigned int gridY = (height + blockY - 1) / blockY;
    unsigned int gridZ = nposes;

    void* kernel_args[] = {&params};

    hipError_t hip_err = hipModuleLaunchKernel(
        static_cast<hipFunction_t>(m_kernel_func),
        gridX, gridY, gridZ,
        blockX, blockY, 1,
        0,
        nullptr,
        kernel_args,
        nullptr
    );

    if (hip_err != hipSuccess) {
        throw std::runtime_error(
            std::string("hipModuleLaunchKernel failed: ") + hipGetErrorString(hip_err));
    }

    hip_err = hipDeviceSynchronize();
    if (hip_err != hipSuccess) {
        throw std::runtime_error(
            std::string("hipDeviceSynchronize failed: ") + hipGetErrorString(hip_err));
    }
}

} // namespace rmagine
