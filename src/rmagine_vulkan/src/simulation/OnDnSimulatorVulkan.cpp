#include "rmagine/simulation/OnDnSimulatorVulkan.hpp"



namespace rmagine
{

void OnDnSimulatorVulkan::setModel(const Memory<OnDnModel, RAM>& sensorMem_ram)
{
    sensorMem_half_ram[0].range = sensorMem_ram[0].range;
    sensorMem_half_ram[0].width = sensorMem_ram[0].width;
    sensorMem_half_ram[0].height = sensorMem_ram[0].height;
    sensorMem_half_ram[0].origs = Memory<Vector3, VULKAN_DEVICE_LOCAL>(sensorMem_ram[0].origs.size());
    sensorMem_half_ram[0].origs = sensorMem_ram[0].origs;
    sensorMem_half_ram[0].dirs = Memory<Vector3, VULKAN_DEVICE_LOCAL>(sensorMem_ram[0].dirs.size());
    sensorMem_half_ram[0].dirs = sensorMem_ram[0].dirs;

    sensorMem = sensorMem_half_ram;
    
    newDimensions.width = sensorMem_ram[0].width;
    newDimensions.height = sensorMem_ram[0].height;
}

} // namespace rmagine
