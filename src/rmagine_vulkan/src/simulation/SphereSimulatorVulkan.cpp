#include "rmagine/simulation/SphereSimulatorVulkan.hpp"



namespace rmagine
{

void SphereSimulatorVulkan::setModel(const Memory<SphericalModel, RAM>& sensorMem_ram)
{
    this->sensorMem = sensorMem_ram;
    newDimensions.width = sensorMem_ram[0].theta.size;
    newDimensions.height = sensorMem_ram[0].phi.size;
}


void SphereSimulatorVulkan::setModel(const SphericalModel& sensor)
{
    Memory<SphericalModel, RAM> sensorMem_ram(1);
    sensorMem_ram[0] = sensor;
    setModel(sensorMem_ram);
}

} // namespace rmagine
