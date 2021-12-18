#ifndef IMAGINE_SIMULATION_ONDN_SIMULATOR_EMBREE_HPP
#define IMAGINE_SIMULATION_ONDN_SIMULATOR_EMBREE_HPP

#include <imagine/map/EmbreeMap.hpp>
#include <imagine/types/Memory.hpp>
#include <imagine/types/sensor_models.h>
#include "SimulationResults.hpp"

namespace imagine
{

class OnDnSimulatorEmbree {
public:
    OnDnSimulatorEmbree(const EmbreeMapPtr map);
    ~OnDnSimulatorEmbree();

    void setTsb(const Memory<Transform, RAM>& Tsb);

    void setModel(const OnDnModel<RAM>& model);
    void setModel(const Memory<OnDnModel<RAM>, RAM>& model);

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
    Memory<OnDnModel<RAM>, RAM> m_model;
};

using OnDnSimulatorEmbreePtr = std::shared_ptr<OnDnSimulatorEmbree>;

} // namespace imagine

#include "OnDnSimulatorEmbree.tcc"

#endif // IMAGINE_SIMULATION_O1DN_SIMULATOR_EMBREE_HPP