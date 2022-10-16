/*
 * Copyright (c) 2022, University Osnabrück
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
 * @brief Simulator Container to construct Embree Simulators more intuitively
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_SIMULATION_SIMULATOR_EMBREE_HPP
#define RMAGINE_SIMULATION_SIMULATOR_EMBREE_HPP

#include "Simulator.hpp"

#include "SphereSimulatorEmbree.hpp"
#include "PinholeSimulatorEmbree.hpp"
#include "O1DnSimulatorEmbree.hpp"
#include "OnDnSimulatorEmbree.hpp"

namespace rmagine
{

// Computing type
struct Embree {

};

template<>
class SimulatorType<SphericalModel, Embree>
{
public:
    using Class = SphereSimulatorEmbree;
    using Ptr = SphereSimulatorEmbreePtr;
};

template<>
class SimulatorType<PinholeModel, Embree>
{
public:
    using Class = PinholeSimulatorEmbree;
    using Ptr = PinholeSimulatorEmbreePtr;
};

template<>
class SimulatorType<O1DnModel, Embree>
{
public:
    using Class = O1DnSimulatorEmbree;
    using Ptr = O1DnSimulatorEmbreePtr;
};

template<>
class SimulatorType<OnDnModel, Embree>
{
public:
    using Class = OnDnSimulatorEmbree;
    using Ptr = OnDnSimulatorEmbreePtr;
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_SIMULATOR_EMBREE_HPP