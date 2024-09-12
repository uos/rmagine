#include "rmagine/simulation/OnDnSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

OnDnSimulatorEmbree::OnDnSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
}

OnDnSimulatorEmbree::OnDnSimulatorEmbree(EmbreeMapPtr map)
:OnDnSimulatorEmbree()
{
    setMap(map);
}

OnDnSimulatorEmbree::~OnDnSimulatorEmbree()
{
    // std::cout << "O1DnSimulatorEmbree - Destructor " << std::endl;
}

void OnDnSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    m_map = map;
}

void OnDnSimulatorEmbree::setTsb(
    const MemoryView<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void OnDnSimulatorEmbree::setTsb(
    const Transform& Tsb)
{
    m_Tsb.resize(1);
    m_Tsb[0] = Tsb;
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