#include "rmagine/simulation/PinholeSimulatorVulkan.hpp"



namespace rmagine
{

void PinholeSimulatorVulkan::setModel(const Memory<PinholeModel, RAM>& sensorMem_ram)
{
    this->sensorMem = sensorMem_ram;
    newDimensions.width = sensorMem_ram[0].width;
    newDimensions.height = sensorMem_ram[0].height;
}


void PinholeSimulatorVulkan::setModel(const PinholeModel& sensor)
{
    Memory<PinholeModel, RAM> sensorMem_ram(1);
    sensorMem_ram[0] = sensor;
    setModel(sensorMem_ram);
}

} // namespace rmagine
