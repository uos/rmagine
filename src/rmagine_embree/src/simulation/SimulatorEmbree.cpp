#include "rmagine/simulation/SimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

  SimulatorEmbree::SimulatorEmbree()
:m_Tsb(1)
{
  m_Tsb[0].setIdentity();
}

SimulatorEmbree::SimulatorEmbree(const EmbreeMapPtr map)
:SimulatorEmbree()
{
  setMap(map);
}

SimulatorEmbree::~SimulatorEmbree()
{
    
}

void SimulatorEmbree::setMap(EmbreeMapPtr map)
{
  m_map = map;
}

void SimulatorEmbree::setTsb(const MemoryView<Transform, RAM>& Tsb)
{
  m_Tsb = Tsb;
}

void SimulatorEmbree::setTsb(const Transform& Tsb)
{
  m_Tsb.resize(1);
  m_Tsb[0] = Tsb;
}

} // namespace rmagine