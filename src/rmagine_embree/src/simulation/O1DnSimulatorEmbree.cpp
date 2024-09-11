#include "rmagine/simulation/O1DnSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

O1DnSimulatorEmbree::O1DnSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
}

O1DnSimulatorEmbree::O1DnSimulatorEmbree(EmbreeMapPtr map)
:O1DnSimulatorEmbree()
{
    setMap(map);
}

O1DnSimulatorEmbree::~O1DnSimulatorEmbree()
{
    
}

void O1DnSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    m_map = map;
}

void O1DnSimulatorEmbree::setTsb(const MemoryView<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void O1DnSimulatorEmbree::setTsb(const Transform& Tsb)
{
    m_Tsb[0] = Tsb;
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