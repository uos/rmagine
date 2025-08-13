#include "rmagine/simulation/SphereSimulatorVulkan.hpp"



namespace rmagine
{

void SphereSimulatorVulkan::setModel(Memory<SphericalModel, RAM>& sensorMem_ram)
{
    this->sensorMem = sensorMem_ram;
    newDimensions.width = sensorMem_ram[0].theta.size;
    newDimensions.height = sensorMem_ram[0].phi.size;
}

} // namespace rmagine
