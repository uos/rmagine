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
 * @brief Contains @link rmagine::O1DnSimulatorEmbree O1DnSimulatorEmbree @endlink
 *
 * @date 03.01.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2021, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_SIMULATION_O1DN_SIMULATOR_EMBREE_HPP
#define RMAGINE_SIMULATION_O1DN_SIMULATOR_EMBREE_HPP

#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/simulation/SimulationResults.hpp>

#include "SimulatorEmbree.hpp"

namespace rmagine
{

/**
 * @brief O1DnModel simulation on CPU via Embree
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
 * EmbreeMapPtr map = import_embree_map("somemesh.ply");
 * // Construct the simulator, that operates on a specific map 
 * O1DnSimulatorEmbree sim(map);
 * 
 * size_t Nposes = 100;
 * 
 * // Inputs
 * Memory<Transform, RAM> T_sensor_to_base(1); // Static transform between sensor and base frame
 * O1DnModel_<RAM> model; // O1DnModel in RAM
 * Memory<Transform, RAM> T_base_to_map(Nposes); // Poses in VRAM
 * // fill data
 * 
 * // set model and sensor to base transform
 * sim.setTsb(T_sensor_to_base);
 * sim.setModel(model);
 * 
 * // Define your desired simulation results.
 * // Possible Elements to Simulate are defined in SimulationResults.hpp
 * using ResT = Bundle<Points<VRAM_CUDA>, Normals<VRAM_CUDA> >;
 * 
 * // Simulate
 * ResT results = sim.simulate<ResT>(T_base_to_map);
 * 
 * // Pass results to other code
 * // Or access them via:
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
class O1DnSimulatorEmbree 
: public SimulatorEmbree
{
public:
  O1DnSimulatorEmbree();
  O1DnSimulatorEmbree(EmbreeMapPtr map);
  ~O1DnSimulatorEmbree();


  void setModel(const MemoryView<O1DnModel_<RAM>, RAM>& model);
  void setModel(const O1DnModel_<RAM>& model);

  inline Memory<O1DnModel_<RAM>, RAM> model() const
  {
      return m_model;
  }

  /**
   * @brief Simulate from one pose
   * 
   * @tparam BundleT 
   * @param Tbm 
   * @param ret 
   */
  template<typename BundleT>
  void simulate(const Transform& Tbm, BundleT& ret) const;

  template<typename BundleT>
  BundleT simulate(const Transform& Tbm) const;

  /**
   * @brief Simulate for multiple poses at once
   * 
   * @tparam BundleT 
   * @param Tbm 
   * @param ret 
   */
  template<typename BundleT>
  void simulate(const MemoryView<Transform, RAM>& Tbm,
      BundleT& ret) const;

  template<typename BundleT>
  void simulate(const MemoryView<const Transform, RAM>& Tbm,
      BundleT& ret) const;

  template<typename BundleT>
  BundleT simulate(const MemoryView<Transform, RAM>& Tbm) const;

  template<typename BundleT>
  BundleT simulate(const MemoryView<const Transform, RAM>& Tbm) const;

protected:
  Memory<O1DnModel_<RAM>, RAM> m_model;
};

using O1DnSimulatorEmbreePtr = std::shared_ptr<O1DnSimulatorEmbree>;

} // namespace rmagine

#include "O1DnSimulatorEmbree.tcc"

#endif // RMAGINE_SIMULATION_O1DN_SIMULATOR_EMBREE_HPP