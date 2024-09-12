#include "rmagine/simulation/SphereSimulatorEmbree.hpp"
#include <limits>


namespace rmagine
{

SphereSimulatorEmbree::SphereSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
    // std::cout << "[SphereSimulatorEmbree::SphereSimulatorEmbree()] constructed." << std::endl;
}

SphereSimulatorEmbree::SphereSimulatorEmbree(EmbreeMapPtr map)
:SphereSimulatorEmbree()
{
    setMap(map);
}

SphereSimulatorEmbree::~SphereSimulatorEmbree()
{
    
}

void SphereSimulatorEmbree::setMap(
    EmbreeMapPtr map)
{
    m_map = map;
}

void SphereSimulatorEmbree::setTsb(
    const MemoryView<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void SphereSimulatorEmbree::setTsb(
    const Transform& Tsb)
{
    m_Tsb.resize(1);
    m_Tsb[0] = Tsb;
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