#ifndef IMAGINE_SIMULATION_SPHERE_SIMULATOR_EMBREE_HPP
#define IMAGINE_SIMULATION_SPHERE_SIMULATOR_EMBREE_HPP

#include <imagine/map/EmbreeMap.hpp>
#include <imagine/types/Memory.hpp>
#include <imagine/types/sensor_models.h>
#include "SimulationResults.hpp"

namespace imagine
{

class SphereSimulatorEmbree {
public:
    SphereSimulatorEmbree(const EmbreeMapPtr mesh);
    ~SphereSimulatorEmbree();

    void setTsb(const Memory<Transform, RAM>& Tsb);
    void setModel(const Memory<SphericalModel, RAM>& model);

    void simulateRanges(
        const Memory<Transform, RAM>& Tbm, 
        Memory<float, RAM>& ranges);

    Memory<float, RAM> simulateRanges(
        const Memory<Transform, RAM>& Tbm);

    void simulateHits(
        const Memory<Transform, RAM>& Tbm, 
        Memory<uint8_t, RAM>& hits);

    Memory<uint8_t, RAM> simulateHits(
        const Memory<Transform, RAM>& Tbm);

    // Generic Version
    template<typename BundleT>
    void simulate(const Memory<Transform, RAM>& Tbm,
        BundleT& ret);

    template<typename BundleT>
    BundleT simulate(const Memory<Transform, RAM>& Tbm);
    
protected:
    const EmbreeMapPtr m_map;
    RTCIntersectContext m_context;

    Memory<Transform, RAM> m_Tsb;
    Memory<SphericalModel, RAM> m_model;
};

using SphereSimulatorEmbreePtr = std::shared_ptr<SphereSimulatorEmbree>;

} // namespace imagine

#include "SphereSimulatorEmbree.tcc"

#endif // IMAGINE_SIMULATION_SPHERE_SIMULATOR_EMBREE_HPP