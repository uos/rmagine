/*
 * Copyright (c) 2021, University Osnabrück. 
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Contains @link rmagine::SphereSimulatorOptix SphereSimulatorOptix @endlink
 *
 * @date 03.01.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2021, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 */

#ifndef RMAGINE_SPHERE_SIMULATOR_OPTIX_HPP
#define RMAGINE_SPHERE_SIMULATOR_OPTIX_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/map/optix/optix_definitions.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>

// Generic
#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/Bundle.hpp>
#include <rmagine/simulation/optix/sim_program_data.h>

#include <cuda_runtime.h>

#include <unordered_map>

#include <rmagine/util/cuda/cuda_definitions.h>

#include <rmagine/util/optix/optix_modules.h>


namespace rmagine {


/**
 * @brief Spherical model simulation on GPU via Optix
 * 
 * Example:
 * 
 * @code{cpp}
 * 
 * #include <rmagine/simulation.h>
 * 
 * using namespace rmagine;
 * 
 * // Import a map
 * OptixMapPtr map = importOptixMap("somemap.ply");
 * // Construct the simulator, that operates on a specific map
 * SphereSimulatorOptix sim(map);
 * 
 * size_t Nposes = 100;
 * 
 * // Inputs
 * Memory<Transform, RAM> T_sensor_to_base(1); // Static transform between sensor and base frame
 * Memory<SphericalModel, RAM> model(1); // SphericalModel in RAM
 * Memory<Transform, RAM> T_base_to_map(Nposes); // Poses in VRAM
 * // fill data
 * 
 * // set model and sensor to base transform
 * sim.setTsb(T_sensor_to_base);
 * sim.setModel(model);
 * 
 * // Upload poses to gpu
 * Memory<Transform, VRAM_CUDA> T_base_to_map_gpu;
 * T_base_to_map_gpu = T_base_to_map;
 * 
 * 
 * // Define your desired simulation results.
 * // Possible Elements to Simulate are defined in SimulationResults.hpp
 * using ResT = Bundle<Points<VRAM_CUDA>, Normals<VRAM_CUDA> >;
 * 
 * // Simulate
 * ResT results = sim.simulate<ResT>(T_base_to_map_gpu);
 * 
 * // Pass results to other cuda code
 * // Or access them via download:
 * Memory<Vector, RAM> points, normals;
 * points = results.points;
 * normals = results.normals;
 * 
 * // Points are of 1D shape.
 * // First Point of i-th Scan
 * int i=0;
 * points[i * model->size() ];
 * 
 * @endcode
 * 
 */
class SphereSimulatorOptix {
public:
    SphereSimulatorOptix();
    SphereSimulatorOptix(OptixMapPtr map);
    SphereSimulatorOptix(OptixScenePtr scene);

    ~SphereSimulatorOptix();

    void setMap(const OptixMapPtr map);

    void setTsb(const Memory<Transform, RAM>& Tsb);
    void setTsb(const Transform& Tsb);

    void setModel(const Memory<SphericalModel, RAM>& model);
    void setModel(const SphericalModel& model);

    void simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm, 
        Memory<float, VRAM_CUDA>& ranges) const;

    Memory<float, VRAM_CUDA> simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm) const;
    /**
     * @brief Simulation of a LiDAR-Sensor in a given mesh
     * 
     * @tparam ResultT Pass disired results via ResultT=Bundle<...>;
     * @param Tbm Transformations between base and map. eg Poses or Particles. In VRAM
     * @return ResultT 
     */
    template<typename BundleT>
    BundleT simulate(
        const Memory<Transform, VRAM_CUDA>& Tbm);

    template<typename BundleT>
    void simulate(
        const Memory<Transform, VRAM_CUDA>& Tbm,
        BundleT& res);

    template<typename BundleT>
    void preBuildProgram();

    // Problems:
    // - a lot of copies
    // 
    // Solutions:
    // - link buffers instead
    // 
    // Example:
    // member Buffer that is partially upgraded
    // member

protected:
    OptixMapPtr m_map;
    CudaStreamPtr m_stream;

    uint32_t m_width;
    uint32_t m_height;
    Memory<Transform, VRAM_CUDA> m_Tsb;
    Memory<SphericalModel, VRAM_CUDA> m_model;

    Memory<SensorModelUnion, VRAM_CUDA> m_model_union;
private:
    void launch(
        const Memory<OptixSimulationDataGeneric, RAM>& mem,
        PipelinePtr program
    );

    std::vector<PipelinePtr> m_programs;
};

using SphereSimulatorOptixPtr = std::shared_ptr<SphereSimulatorOptix>;

} // namespace rmagine

#include "SphereSimulatorOptix.tcc"

#endif // RMAGINE_OPTIX_SIMULATOR_HPP