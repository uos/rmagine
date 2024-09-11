#include "rmagine/simulation/PinholeSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

PinholeSimulatorEmbree::PinholeSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
}

PinholeSimulatorEmbree::PinholeSimulatorEmbree(const EmbreeMapPtr map)
:PinholeSimulatorEmbree()
{
    setMap(map);
}

PinholeSimulatorEmbree::~PinholeSimulatorEmbree()
{
    
}

void PinholeSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    m_map = map;
}

void PinholeSimulatorEmbree::setTsb(const MemoryView<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void PinholeSimulatorEmbree::setTsb(const Transform& Tsb)
{
    m_Tsb.resize(1);
    m_Tsb[0] = Tsb;
}

void PinholeSimulatorEmbree::setModel(const MemoryView<PinholeModel, RAM>& model)
{
    m_model = model;
}

void PinholeSimulatorEmbree::setModel(const PinholeModel& model)
{
    m_model.resize(1);
    m_model[0] = model;
}

} // namespace rmagine