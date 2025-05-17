#include "rmagine/simulation/OnDnSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

OnDnSimulatorEmbree::OnDnSimulatorEmbree()
:SimulatorEmbree()
,m_model(1)
{
  
}

OnDnSimulatorEmbree::OnDnSimulatorEmbree(EmbreeMapPtr map)
:SimulatorEmbree(map)
,m_model(1)
{
  
}

OnDnSimulatorEmbree::~OnDnSimulatorEmbree()
{
  // std::cout << "O1DnSimulatorEmbree - Destructor " << std::endl;
}

void OnDnSimulatorEmbree::setModel(
    const OnDnModel_<RAM>& model)
{
  m_model[0] = model;
}

void OnDnSimulatorEmbree::setModel(
    const MemoryView<OnDnModel_<RAM>, RAM>& model)
{
  m_model->width = model->width;
  m_model->height = model->height;
  m_model->range = model->range;
  m_model->origs = model->origs;
  m_model->dirs = model->dirs;
}

} // namespace rmagine