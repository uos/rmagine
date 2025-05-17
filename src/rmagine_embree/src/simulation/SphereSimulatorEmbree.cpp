#include "rmagine/simulation/SphereSimulatorEmbree.hpp"
#include <limits>


namespace rmagine
{

SphereSimulatorEmbree::SphereSimulatorEmbree()
:SimulatorEmbree()
,m_model(1)
{
  
}

SphereSimulatorEmbree::SphereSimulatorEmbree(EmbreeMapPtr map)
:SimulatorEmbree(map)
,m_model(1)
{
  
}

SphereSimulatorEmbree::~SphereSimulatorEmbree()
{
    
}

void SphereSimulatorEmbree::setModel(
    const MemoryView<SphericalModel, RAM>& model)
{
  m_model = model;
}

void SphereSimulatorEmbree::setModel(
    const SphericalModel& model)
{
  m_model.resize(1);
  m_model[0] = model;
}

} // namespace rmagine