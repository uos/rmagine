/**
 * @file OnDnSimulatorHiprt.hpp
 * @brief OnDn (N origins, N directions) ray simulator using HIPRT
 *
 * Mirrors OnDnSimulatorOptix but uses HIPRT for ray tracing.
 * Used for sensors like depth camera arrays where each ray has its own
 * origin and direction.
 */

#ifndef RMAGINE_SIMULATION_ONDN_SIMULATOR_HIPRT_HPP
#define RMAGINE_SIMULATION_ONDN_SIMULATOR_HIPRT_HPP

#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>
#include <rmagine/map/hiprt/HiprtScene.hpp>
#include <rmagine/simulation/hiprt/sim_program_data.h>

namespace rmagine
{

class OnDnSimulatorHiprt;
using OnDnSimulatorHiprtPtr = std::shared_ptr<OnDnSimulatorHiprt>;

/**
 * @brief OnDn sensor simulator using HIPRT ray tracing
 *
 * N origins, N directions. Each ray has its own origin and direction
 * (e.g., a depth camera array).
 */
class OnDnSimulatorHiprt
{
public:
    OnDnSimulatorHiprt();
    explicit OnDnSimulatorHiprt(HiprtScenePtr scene);
    ~OnDnSimulatorHiprt();

    void setScene(HiprtScenePtr scene);

    void setModel(const OnDnModel& model);

    void setTsb(const Transform& Tsb);
    void setTsb(const Memory<Transform, RAM>& Tsb);

    Memory<float, VRAM_CUDA> simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    Memory<uint8_t, VRAM_CUDA> simulateHits(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    Memory<Point, VRAM_CUDA> simulatePoints(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    void simulate(
        const Memory<Transform, VRAM_CUDA>& Tbm,
        HiprtSimulationData& data);

private:
    HiprtScenePtr m_scene;

    // OnDn model has device memory for origins and directions
    Memory<Vector, VRAM_CUDA> m_origs;  // device copy of origins
    Memory<Vector, VRAM_CUDA> m_dirs;   // device copy of directions
    Memory<HiprtOnDnModelDevice, VRAM_CUDA> m_model_device;
    Memory<HiprtSensorModelUnion, VRAM_CUDA> m_model_union;
    unsigned int m_width = 0;
    unsigned int m_height = 0;

    Memory<Transform, VRAM_CUDA> m_Tsb;

    void* m_kernel_module = nullptr;
    void* m_kernel_func = nullptr;

    void ensureKernel();

    void launchKernel(
        const HiprtSimulationData& data,
        unsigned int width,
        unsigned int height,
        unsigned int nposes);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_ONDN_SIMULATOR_HIPRT_HPP
