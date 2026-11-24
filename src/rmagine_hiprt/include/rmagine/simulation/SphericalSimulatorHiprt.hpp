/**
 * @file SphericalSimulatorHiprt.hpp
 * @brief Spherical (lidar) ray simulator using HIPRT
 *
 * Mirrors SphereSimulatorOptix but uses HIPRT for ray tracing.
 */

#ifndef RMAGINE_SIMULATION_SPHERICAL_SIMULATOR_HIPRT_HPP
#define RMAGINE_SIMULATION_SPHERICAL_SIMULATOR_HIPRT_HPP

#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/types.h>
#include <rmagine/map/hiprt/HiprtScene.hpp>
#include <rmagine/simulation/hiprt/sim_program_data.h>

namespace rmagine
{

class SphericalSimulatorHiprt;
using SphericalSimulatorHiprtPtr = std::shared_ptr<SphericalSimulatorHiprt>;

/**
 * @brief Spherical sensor simulator using HIPRT ray tracing
 *
 * Traces rays from a spherical sensor model (lidar) against a HIPRT scene.
 * Ray directions are computed from spherical coordinates (phi, theta).
 */
class SphericalSimulatorHiprt
{
public:
    SphericalSimulatorHiprt();
    explicit SphericalSimulatorHiprt(HiprtScenePtr scene);
    ~SphericalSimulatorHiprt();

    void setScene(HiprtScenePtr scene);

    void setModel(const SphericalModel& model);
    void setModel(const Memory<SphericalModel, RAM>& model);

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

    Memory<SphericalModel, VRAM_CUDA> m_model;
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

#endif // RMAGINE_SIMULATION_SPHERICAL_SIMULATOR_HIPRT_HPP
