/**
 * @file O1DnSimulatorHiprt.hpp
 * @brief O1Dn (One origin, N directions) ray simulator using HIPRT
 *
 * Mirrors O1DnSimulatorOptix but uses HIPRT for ray tracing.
 * Used for sensors like planar lidars where all rays share a common origin.
 */

#ifndef RMAGINE_SIMULATION_O1DN_SIMULATOR_HIPRT_HPP
#define RMAGINE_SIMULATION_O1DN_SIMULATOR_HIPRT_HPP

#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>
#include <rmagine/map/hiprt/HiprtScene.hpp>
#include <rmagine/simulation/hiprt/sim_program_data.h>

namespace rmagine
{

class O1DnSimulatorHiprt;
using O1DnSimulatorHiprtPtr = std::shared_ptr<O1DnSimulatorHiprt>;

/**
 * @brief O1Dn sensor simulator using HIPRT ray tracing
 *
 * One origin, N directions. All rays share a common origin but have
 * different directions (e.g., a planar lidar scan).
 */
class O1DnSimulatorHiprt
{
public:
    O1DnSimulatorHiprt();
    explicit O1DnSimulatorHiprt(HiprtScenePtr scene);
    ~O1DnSimulatorHiprt();

    void setScene(HiprtScenePtr scene);

    void setModel(const O1DnModel& model);

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

    // O1Dn model has device memory for directions
    Memory<Vector, VRAM_CUDA> m_dirs;  // device copy of directions
    Memory<HiprtO1DnModelDevice, VRAM_CUDA> m_model_device;
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

#endif // RMAGINE_SIMULATION_O1DN_SIMULATOR_HIPRT_HPP
