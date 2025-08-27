#include "rmagine/simulation/OnDnSimulatorVulkan.hpp"



namespace rmagine
{

void OnDnSimulatorVulkan::setModel(const Memory<OnDnModel, RAM>& sensorMem_ram)
{
    sensorMem = sensorMem_ram;

    origs.resize(sensorMem_ram[0].origs.size());
    origs = sensorMem_ram[0].origs;

    dirs.resize(sensorMem_ram[0].dirs.size());
    dirs = sensorMem_ram[0].dirs;
    
    newDimensions.width = sensorMem_ram[0].width;
    newDimensions.height = sensorMem_ram[0].height;
}

void OnDnSimulatorVulkan::updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem)
{
    if(previousAddresses.tbmAndSensorSpecificAddresses.tbmAddress   != tbmMem.getBuffer()->getBufferDeviceAddress() ||
       previousAddresses.tbmAndSensorSpecificAddresses.origsAddress != origs.getBuffer()->getBufferDeviceAddress()  ||
       previousAddresses.tbmAndSensorSpecificAddresses.dirsAddress  != dirs.getBuffer()->getBufferDeviceAddress())
    {
        Memory<VulkanTbmAndSensorSpecificAddresses, RAM> origsDirsAndTransformsMem_ram(1);
        origsDirsAndTransformsMem_ram[0].tbmAddress   = tbmMem.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].origsAddress = origs.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].dirsAddress  = dirs.getBuffer()->getBufferDeviceAddress();

        previousAddresses.tbmAndSensorSpecificAddresses = origsDirsAndTransformsMem_ram[0];

        origsDirsAndTransformsMem = origsDirsAndTransformsMem_ram;
    }
}

} // namespace rmagine
