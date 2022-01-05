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
 * @brief Contains @link imagine::SphereSimulatorOptix SphereSimulatorOptix @endlink
 *
 * @date 03.01.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2021, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 */

#ifndef IMAGINE_SPHERE_SIMULATOR_OPTIX_HPP
#define IMAGINE_SPHERE_SIMULATOR_OPTIX_HPP

#include <optix.h>

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>
#include <imagine/types/MemoryCuda.hpp>
#include <imagine/types/sensor_models.h>

// Generic
#include <imagine/simulation/SimulationResults.hpp>
#include <imagine/types/Bundle.hpp>
#include <imagine/simulation/optix/OptixSimulationData.hpp>
#include <imagine/simulation/optix/OptixProgramMap.hpp>

#include <cuda_runtime.h>

#include <unordered_map>


namespace imagine {


/**
 * @brief Spherical model simulation on GPU via Optix
 * 
 * Example:
 * 
 * @code{cpp}
 * 
 * #include <imagine/simulation.h>
 * 
 * using namespace imagine;
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
    SphereSimulatorOptix(OptixMapPtr map);

    ~SphereSimulatorOptix();

    void setTsb(const Memory<Transform, RAM>& Tsb);

    void setModel(const Memory<SphericalModel, RAM>& model);

    void simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm, 
        Memory<float, VRAM_CUDA>& ranges) const;

    Memory<float, VRAM_CUDA> simulateRanges(
        const Memory<Transform, VRAM_CUDA>& Tbm) const;

    void simulateNormals(
        const Memory<Transform, VRAM_CUDA>& Tbm, 
        Memory<Vector, VRAM_CUDA>& normals) const;

    Memory<Vector, VRAM_CUDA> simulateNormals(
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
    cudaStream_t m_stream;

    uint32_t m_width;
    uint32_t m_height;
    Memory<Transform, VRAM_CUDA> m_Tsb;
    Memory<SphericalModel, VRAM_CUDA> m_model;

private:
    std::vector<OptixProgramPtr> m_programs;

    std::unordered_map<OptixSimulationDataGenericSphere, OptixProgramPtr> m_generic_programs;
};

using SphereSimulatorOptixPtr = std::shared_ptr<SphereSimulatorOptix>;

} // namespace imagine

#include "SphereSimulatorOptix.tcc"

#endif // IMAGINE_OPTIX_SIMULATOR_HPP