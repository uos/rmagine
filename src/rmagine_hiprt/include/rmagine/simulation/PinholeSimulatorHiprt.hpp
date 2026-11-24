/**
 * @file PinholeSimulatorHiprt.hpp
 * @brief Pinhole camera ray simulator using HIPRT
 *
 * Mirrors PinholeSimulatorOptix but uses HIPRT for ray tracing.
 */

#ifndef RMAGINE_SIMULATION_PINHOLE_SIMULATOR_HIPRT_HPP
#define RMAGINE_SIMULATION_PINHOLE_SIMULATOR_HIPRT_HPP

#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>
#include <rmagine/map/hiprt/HiprtScene.hpp>
#include <rmagine/simulation/hiprt/sim_program_data.h>

namespace rmagine
{

class PinholeSimulatorHiprt;
using PinholeSimulatorHiprtPtr = std::shared_ptr<PinholeSimulatorHiprt>;

/**
 * @brief Pinhole camera simulator using HIPRT ray tracing
 *
 * Traces rays from a pinhole camera model against a HIPRT scene
 * and returns hit results (ranges, points, normals, etc.).
 */
class PinholeSimulatorHiprt
{
public:
    PinholeSimulatorHiprt();
    explicit PinholeSimulatorHiprt(HiprtScenePtr scene);
    ~PinholeSimulatorHiprt();

    /**
     * @brief Set the scene to trace against
     */
    void setScene(HiprtScenePtr scene);

    /**
     * @brief Set the sensor model
     */
    void setModel(const PinholeModel& model);
    void setModel(const Memory<PinholeModel, RAM>& model);

    /**
     * @brief Set the sensor-to-body transform
     */
    void setTsb(const Transform& Tsb);
    void setTsb(const Memory<Transform, RAM>& Tsb);

    /**
     * @brief Simulate and get ranges
     *
     * @param Tbm Body-to-map poses (one per simulation)
     * @return Range values for each ray
     */
    Memory<float, VRAM_CUDA> simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    /**
     * @brief Simulate and get hits
     *
     * @param Tbm Body-to-map poses
     * @return Hit flags (1=hit, 0=miss) for each ray
     */
    Memory<uint8_t, VRAM_CUDA> simulateHits(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    /**
     * @brief Simulate and get points in sensor frame
     *
     * @param Tbm Body-to-map poses
     * @return 3D points for each ray (NaN for misses)
     */
    Memory<Point, VRAM_CUDA> simulatePoints(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    /**
     * @brief Simulate with full output control
     *
     * @param Tbm Body-to-map poses
     * @param data Simulation data with output flags and buffers
     */
    void simulate(
        const Memory<Transform, VRAM_CUDA>& Tbm,
        HiprtSimulationData& data);

private:
    HiprtScenePtr m_scene;

    // Sensor model
    Memory<PinholeModel, VRAM_CUDA> m_model;
    Memory<HiprtSensorModelUnion, VRAM_CUDA> m_model_union;
    unsigned int m_width = 0;
    unsigned int m_height = 0;

    // Sensor-to-body transform
    Memory<Transform, VRAM_CUDA> m_Tsb;

    // JIT-compiled trace kernel handle
    void* m_kernel_module = nullptr;
    void* m_kernel_func = nullptr;

    /**
     * @brief Ensure the trace kernel is compiled
     */
    void ensureKernel();

    /**
     * @brief Launch the trace kernel
     */
    void launchKernel(
        const HiprtSimulationData& data,
        unsigned int width,
        unsigned int height,
        unsigned int nposes);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_PINHOLE_SIMULATOR_HIPRT_HPP
