#ifndef IMAGINE_SIMULATION_PINHOLE_SIMULATOR_EMBREE_HPP
#define IMAGINE_SIMULATION_PINHOLE_SIMULATOR_EMBREE_HPP

#include <imagine/map/EmbreeMap.hpp>
#include <imagine/types/Memory.hpp>
#include <imagine/types/sensor_models.h>
#include "SimulationResults.hpp"

namespace imagine
{

class PinholeSimulatorEmbree {
public:
    PinholeSimulatorEmbree(const EmbreeMapPtr map);
    ~PinholeSimulatorEmbree();

    void setTsb(const Memory<Transform, RAM>& Tsb);
    void setModel(const Memory<PinholeModel, RAM>& model);

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
    Memory<PinholeModel, RAM> m_model;
};

using PinholeSimulatorEmbreePtr = std::shared_ptr<PinholeSimulatorEmbree>;

} // namespace imagine

#include "PinholeSimulatorEmbree.tcc"

#endif // IMAGINE_SIMULATION_PINHOLE_SIMULATOR_EMBREE_HPP