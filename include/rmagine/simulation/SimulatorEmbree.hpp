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