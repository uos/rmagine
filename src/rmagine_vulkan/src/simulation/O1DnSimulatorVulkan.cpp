#include "rmagine/simulation/O1DnSimulatorVulkan.hpp"



namespace rmagine
{

void O1DnSimulatorVulkan::setModel(const Memory<O1DnModel, RAM>& sensorMem_ram)
{
    sensorMem = sensorMem_ram;

    dirs.resize(sensorMem_ram[0].dirs.size());
    dirs = sensorMem_ram[0].dirs;
    
    newDimensions.width = sensorMem_ram[0].width;
    newDimensions.height = sensorMem_ram[0].height;
}




void O1DnSimulatorVulkan::updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem)
{
    if(previousAddresses.tbmAndSensorSpecificAddresses.tbmAddress   != tbmMem.getBuffer()->getBufferDeviceAddress() ||
       previousAddresses.tbmAndSensorSpecificAddresses.origsAddress != 0 ||
       previousAddresses.tbmAndSensorSpecificAddresses.dirsAddress  != dirs.getBuffer()->getBufferDeviceAddress())
    {
        Memory<VulkanTbmAndSensorSpecificAddresses, RAM> origsDirsAndTransformsMem_ram(1);
        origsDirsAndTransformsMem_ram[0].tbmAddress   = tbmMem.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].origsAddress = 0;
        origsDirsAndTransformsMem_ram[0].dirsAddress  = dirs.getBuffer()->getBufferDeviceAddress();

        previousAddresses.tbmAndSensorSpecificAddresses = origsDirsAndTransformsMem_ram[0];

        origsDirsAndTransformsMem = origsDirsAndTransformsMem_ram;
    }
}

} // namespace rmagine
