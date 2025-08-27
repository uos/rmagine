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

void OnDnSimulatorVulkan::updateAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram)
{
    if(previousBuffers.resultsAddresses.hitsAddress        != resultsMem_ram[0].hitsAddress        ||
       previousBuffers.resultsAddresses.rangesAddress      != resultsMem_ram[0].rangesAddress      ||
       previousBuffers.resultsAddresses.pointsAddress      != resultsMem_ram[0].pointsAddress      ||
       previousBuffers.resultsAddresses.normalsAddress     != resultsMem_ram[0].normalsAddress     ||
       previousBuffers.resultsAddresses.primitiveIdAddress != resultsMem_ram[0].primitiveIdAddress ||
       previousBuffers.resultsAddresses.instanceIdAddress  != resultsMem_ram[0].instanceIdAddress  ||
       previousBuffers.resultsAddresses.geometryIdAddress  != resultsMem_ram[0].geometryIdAddress)
    {
        previousBuffers.resultsAddresses = resultsMem_ram[0];

        resultsMem = resultsMem_ram;
    }

    if(previousBuffers.origsDirsAndTransformsAddresses.tbmAddress   != tbmMem.getBuffer()->getBufferDeviceAddress() ||
       previousBuffers.origsDirsAndTransformsAddresses.origsAddress != origs.getBuffer()->getBufferDeviceAddress()  ||
       previousBuffers.origsDirsAndTransformsAddresses.dirsAddress  != dirs.getBuffer()->getBufferDeviceAddress())
    {
        Memory<VulkanOrigsDirsAndTransformsData, RAM> origsDirsAndTransformsMem_ram(1);
        origsDirsAndTransformsMem_ram[0].tbmAddress   = tbmMem.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].origsAddress = origs.getBuffer()->getBufferDeviceAddress();
        origsDirsAndTransformsMem_ram[0].dirsAddress  = dirs.getBuffer()->getBufferDeviceAddress();

        previousBuffers.origsDirsAndTransformsAddresses = origsDirsAndTransformsMem_ram[0];

        origsDirsAndTransformsMem = origsDirsAndTransformsMem_ram;
    }
}

} // namespace rmagine
