#include "rmagine/simulation/O1DnSimulatorVulkan.hpp"



namespace rmagine
{

void O1DnSimulatorVulkan::setModel(const Memory<O1DnModel, RAM>& sensorMem_ram)
{
    sensorMem_half_ram[0].range = sensorMem_ram[0].range;
    sensorMem_half_ram[0].width = sensorMem_ram[0].width;
    sensorMem_half_ram[0].height = sensorMem_ram[0].height;
    sensorMem_half_ram[0].orig = sensorMem_ram[0].orig;
    sensorMem_half_ram[0].dirs = Memory<Vector3, VULKAN_DEVICE_LOCAL>(sensorMem_ram[0].dirs.size());
    sensorMem_half_ram[0].dirs = sensorMem_ram[0].dirs;

    sensorMem = sensorMem_half_ram;
    
    newDimensions.width = sensorMem_ram[0].width;
    newDimensions.height = sensorMem_ram[0].height;
}

} // namespace rmagine
