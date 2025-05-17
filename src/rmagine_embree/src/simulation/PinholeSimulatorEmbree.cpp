#include "rmagine/simulation/PinholeSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

PinholeSimulatorEmbree::PinholeSimulatorEmbree()
:SimulatorEmbree()
,m_model(1)
{
  m_Tsb[0].setIdentity();
}

PinholeSimulatorEmbree::PinholeSimulatorEmbree(const EmbreeMapPtr map)
:SimulatorEmbree(map)
,m_model(1)
{
  
}

PinholeSimulatorEmbree::~PinholeSimulatorEmbree()
{
    
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