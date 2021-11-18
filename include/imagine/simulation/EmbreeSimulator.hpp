#ifndef IMAGINE_EMBREE_SIMULATOR_HPP
#define IMAGINE_EMBREE_SIMULATOR_HPP

#include <imagine/map/EmbreeMap.hpp>
#include <imagine/types/Memory.hpp>
#include <imagine/types/sensor_models.h>
#include "SimulationResults.hpp"

namespace imagine
{

class EmbreeSimulator {
public:
    EmbreeSimulator(const EmbreeMapPtr mesh);
    ~EmbreeSimulator();

    using MEM = RAM;

    void setTsb(const Memory<Transform, RAM>& Tsb);
    void setModel(const Memory<LiDARModel, RAM>& model);

    void simulateRanges(
        const Memory<Transform, RAM>& Tbm, 
        Memory<float, RAM>& ranges);

    Memory<float, RAM> simulateRanges(
        const Memory<Transform, RAM>& Tbm);

    void simulateIds(
        const Memory<Transform, RAM>& Tbm, 
        Memory<MeshFace, RAM>& ids);

    Memory<MeshFace, RAM> simulateIds(
        const Memory<Transform, RAM>& Tbm);
    
protected:
    const EmbreeMapPtr m_map;
    RTCIntersectContext m_context;

    Memory<Transform, RAM> m_Tsb;
    Memory<LiDARModel, RAM> m_model;
};

using EmbreeSimulatorPtr = std::shared_ptr<EmbreeSimulator>;

} // namespace imagine

#endif // IMAGINE_EMBREE_SIMULATOR_HPP