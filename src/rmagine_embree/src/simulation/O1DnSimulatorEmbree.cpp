#include "rmagine/simulation/O1DnSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

O1DnSimulatorEmbree::O1DnSimulatorEmbree()
:SimulatorEmbree()
,m_model(1)
{
  
}

O1DnSimulatorEmbree::O1DnSimulatorEmbree(EmbreeMapPtr map)
:SimulatorEmbree(map)
,m_model(1)
{
  
}

O1DnSimulatorEmbree::~O1DnSimulatorEmbree()
{
    
}

void O1DnSimulatorEmbree::setModel(const O1DnModel_<RAM>& model)
{
   m_model[0] = model;
}

void O1DnSimulatorEmbree::setModel(
    const MemoryView<O1DnModel_<RAM>, RAM>& model)
{
  m_model->width = model->width;
  m_model->height = model->height;
  m_model->range = model->range;
  m_model->orig = model->orig;
  m_model->dirs = model->dirs;
}

} // namespace rmagine