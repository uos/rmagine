/**
 * @file PinholeSimulatorHiprt.cpp
 * @brief Pinhole simulator implementation using HIPRT
 *
 * JIT-compiles a trace kernel using hiprtBuildTraceKernels and launches
 * it via the HIP driver API (hipModuleLaunchKernel).
 *
 * NOTE: Orochi's hipew redeclares HIP driver symbols and conflicts with
 * hip_runtime.h if both are included. To avoid this conflict, we do NOT
 * include Orochi directly. Instead, we use the HIP driver API from
 * hip_runtime.h for kernel launch, since hiprtApiFunction is compatible
 * with hipFunction_t.
 */

#include "rmagine/simulation/PinholeSimulatorHiprt.hpp"

// Include HIPRT (does not pull in hip_runtime.h, only hiprt_types.h)
#include <hiprt/hiprt.h>

// Include HIP runtime for kernel launch
#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace rmagine
{

// Read kernel source file
static std::string readKernelSource(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel source: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Kernel parameters matching the JIT kernel's PinholeTraceParams
// This must be POD and match the kernel's struct exactly
struct PinholeTraceParamsHost
{
    // Pointers are void* here because they're device pointers
    const void* Tsb;
    const void* Tbm;
    unsigned int Nposes;

    const void* model;

    // hiprtGeometry handle (opaque pointer)
    void* geom;

    // Mesh data for normals
    const void* face_normals;
    unsigned int n_meshes;

    // Output control
    bool computeHits;
    bool computeRanges;
    bool computePoints;
    bool computeNormals;

    // Output buffers
    unsigned char* hits;
    float* ranges;
    void* points;
    void* normals;
};

PinholeSimulatorHiprt::PinholeSimulatorHiprt()
    : m_Tsb(1)
{
    // Initialize with identity transform
    Memory<Transform, RAM> I(1);
    I[0].setIdentity();
    m_Tsb = I;
}

PinholeSimulatorHiprt::PinholeSimulatorHiprt(HiprtScenePtr scene)
    : PinholeSimulatorHiprt()
{
    setScene(scene);
}

PinholeSimulatorHiprt::~PinholeSimulatorHiprt()
{
    // Note: We do NOT unload the module here. HIPRT manages the module lifetime
    // internally through its JIT cache, and the module is an Orochi (oroModule)
    // handle, not a HIP (hipModule_t) handle. Attempting to unload with
    // hipModuleUnload causes a symbol lookup failure crash.
    m_kernel_module = nullptr;
    m_kernel_func = nullptr;
}

void PinholeSimulatorHiprt::setScene(HiprtScenePtr scene)
{
    m_scene = scene;
    // Invalidate kernel reference if scene changes (geometry handle changes)
    // We don't unload the module (HIPRT caches it)
    m_kernel_func = nullptr;
}

void PinholeSimulatorHiprt::setModel(const PinholeModel& model)
{
    Memory<PinholeModel, RAM> tmp(1);
    tmp[0] = model;
    setModel(tmp);
}

void PinholeSimulatorHiprt::setModel(const Memory<PinholeModel, RAM>& model)
{
    m_width = model[0].width;
    m_height = model[0].height;
    m_model = model;

    // Create model union
    Memory<HiprtSensorModelUnion, RAM> model_union(1);
    model_union[0].pinhole = m_model.raw();
    m_model_union = model_union;
}

void PinholeSimulatorHiprt::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}

void PinholeSimulatorHiprt::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

Memory<float, VRAM_CUDA> PinholeSimulatorHiprt::simulateRanges(
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

Memory<uint8_t, VRAM_CUDA> PinholeSimulatorHiprt::simulateHits(
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

Memory<Point, VRAM_CUDA> PinholeSimulatorHiprt::simulatePoints(
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

void PinholeSimulatorHiprt::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    HiprtSimulationData& data)
{
    if (!m_scene) {
        throw std::runtime_error("PinholeSimulatorHiprt: no scene set");
    }

    ensureKernel();

    // Fill data structure
    data.Tsb = m_Tsb.raw();
    data.Tbm = Tbm.raw();
    data.Nposes = Tbm.size();
    data.model_type = 0;  // Pinhole
    data.model = m_model_union.raw();
    data.geom_handle = m_scene->handle();
    data.scene_data = &m_scene->sceneData();

    launchKernel(data, m_width, m_height, Tbm.size());
}

void PinholeSimulatorHiprt::ensureKernel()
{
    if (m_kernel_func) {
        return;  // Already compiled
    }

    // Get HIPRT path from environment
    const char* hiprt_path_env = std::getenv("HIPRT_PATH");
    if (!hiprt_path_env) {
        throw std::runtime_error(
            "HIPRT_PATH environment variable not set. "
            "Set it to the HIPRT SDK root (e.g., /path/to/HIPRT).");
    }
    std::string hiprt_path = hiprt_path_env;

    // Read the kernel source
    std::string kernel_path = hiprt_path + "/hiprt/hiprt_device.h";
    std::string rmagine_kernel_src_path;

    // Try to find pinhole_trace_kernel.h relative to rmagine
    // This assumes RMAGINE_HIPRT_KERNEL_PATH is set or falls back to install location
    const char* kernel_src_env = std::getenv("RMAGINE_HIPRT_KERNEL_PATH");
    if (kernel_src_env) {
        rmagine_kernel_src_path = std::string(kernel_src_env) + "/pinhole_trace_kernel.h";
    } else {
        // Default: assume kernel is in the include path
        rmagine_kernel_src_path = "pinhole_trace_kernel.h";
    }

    // Build the kernel source string with necessary includes
    // The kernel source is embedded as a string since hiprtBuildTraceKernels
    // expects inline source, not a file path
    std::string kernel_src = R"(
#include <hiprt/hiprt_device.h>

// Math types matching rmagine (must have identical layout!)
// See rmagine_core/include/rmagine/math/types/*.hpp
struct Vector3f { float x, y, z; };
struct Quaternionf { float x, y, z, w; };
struct Transform3f {
    Quaternionf R;
    Vector3f t;
    unsigned int stamp;  // rmagine::Transform_ has this field
};

// PinholeModel layout matches rmagine::PinholeModel exactly:
// width, height, range{min,max}, f[2], c[2]
struct PinholeModelDev {
    unsigned int width;
    unsigned int height;
    float range_min;
    float range_max;
    float f_x;
    float f_y;
    float c_x;
    float c_y;
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

struct PinholeTraceParams {
    const Transform3f* Tsb;
    const Transform3f* Tbm;
    unsigned int Nposes;
    const PinholeModelDev* model;
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

extern "C" __global__ void pinhole_trace_kernel(PinholeTraceParams params)
{
    const unsigned int hid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int vid = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int pid = blockIdx.z;

    const PinholeModelDev* model = params.model;
    if (hid >= model->width || vid >= model->height || pid >= params.Nposes) return;

    const unsigned int loc_id = vid * model->width + hid;
    const unsigned int glob_id = pid * (model->width * model->height) + loc_id;

    // Ray direction in sensor frame
    Vector3f ray_dir_s;
    ray_dir_s.x = (float(hid) - model->c_x) / model->f_x;
    ray_dir_s.y = (float(vid) - model->c_y) / model->f_y;
    ray_dir_s.z = 1.0f;
    ray_dir_s = normalize3(ray_dir_s);

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
    ray.minT = model->range_min;
    ray.maxT = model->range_max;

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
        if (params.computeRanges && params.ranges) params.ranges[glob_id] = model->range_max + 1.0f;
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

    // JIT compile the kernel using HIPRT
    hiprtContext ctx = m_scene->context()->handle();

    const char* func_names[] = {"pinhole_trace_kernel"};
    hiprtApiFunction funcs[1];
    hiprtApiModule module = nullptr;

    // Include the HIPRT headers
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
        1,                      // numFunctions
        func_names,
        kernel_src.c_str(),
        "pinhole_trace",        // moduleName
        0,                      // numHeaders
        nullptr,                // headers
        nullptr,                // includeNames
        (uint32_t)options.size(),
        options.data(),
        0,                      // numGeomTypes (no custom intersection)
        0,                      // numRayTypes (no custom intersection)
        nullptr,                // funcNameSets
        funcs,
        &module,
        true                    // cache
    );

    if (err != hiprtSuccess) {
        throw std::runtime_error(
            "hiprtBuildTraceKernels failed with error " + std::to_string(err));
    }

    m_kernel_module = module;
    m_kernel_func = reinterpret_cast<void*>(funcs[0]);

    std::cout << "[PinholeSimulatorHiprt] JIT kernel compiled successfully" << std::endl;
}

void PinholeSimulatorHiprt::launchKernel(
    const HiprtSimulationData& data,
    unsigned int width,
    unsigned int height,
    unsigned int nposes)
{
    if (!m_kernel_func) {
        throw std::runtime_error("Kernel not compiled");
    }

    // Build the kernel parameters struct
    // This must match PinholeTraceParams in the kernel exactly
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
    params.model = data.model->pinhole;
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

    // Grid/block dimensions
    const unsigned int blockX = 16, blockY = 16;
    unsigned int gridX = (width + blockX - 1) / blockX;
    unsigned int gridY = (height + blockY - 1) / blockY;
    unsigned int gridZ = nposes;

    // Launch kernel using HIP driver API
    // hiprtApiFunction (void*) is compatible with hipFunction_t
    void* kernel_args[] = {&params};

    hipError_t hip_err = hipModuleLaunchKernel(
        static_cast<hipFunction_t>(m_kernel_func),
        gridX, gridY, gridZ,
        blockX, blockY, 1,
        0,          // shared mem bytes
        nullptr,    // stream (default)
        kernel_args,
        nullptr     // extra
    );

    if (hip_err != hipSuccess) {
        throw std::runtime_error(
            std::string("hipModuleLaunchKernel failed: ") + hipGetErrorString(hip_err));
    }

    // Synchronize to ensure kernel completion
    hip_err = hipDeviceSynchronize();
    if (hip_err != hipSuccess) {
        throw std::runtime_error(
            std::string("hipDeviceSynchronize failed: ") + hipGetErrorString(hip_err));
    }
}

} // namespace rmagine
