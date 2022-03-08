#ifndef RMAGINE_SIMULATION_SIMULATOR_OPTIX_HPP
#define RMAGINE_SIMULATION_SIMULATOR_OPTIX_HPP

#include "Simulator.hpp"

#include "SphereSimulatorOptix.hpp"
#include "PinholeSimulatorOptix.hpp"
#include "O1DnSimulatorOptix.hpp"
#include "OnDnSimulatorOptix.hpp"


namespace rmagine
{

struct Optix {

};

template<>
class SimulatorType<SphericalModel, Optix>
{
public:
    using Class = SphereSimulatorOptix;
    using Ptr = SphereSimulatorOptixPtr;
};

template<>
class SimulatorType<PinholeModel, Optix>
{
public:
    using Class = PinholeSimulatorOptix;
    using Ptr = PinholeSimulatorOptixPtr;
};

template<>
class SimulatorType<O1DnModel, Optix>
{
public:
    using Class = O1DnSimulatorOptix;
    using Ptr = O1DnSimulatorOptixPtr;
};

template<>
class SimulatorType<OnDnModel, Optix>
{
public:
    using Class = OnDnSimulatorOptix;
    using Ptr = OnDnSimulatorOptixPtr;
};

} // namespace rmagine


#endif // RMAGINE_SIMULATION_SIMULATOR_OPTIX_HPP
