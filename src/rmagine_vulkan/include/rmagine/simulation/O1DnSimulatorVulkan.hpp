#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "SimulatorVulkan.hpp"



namespace rmagine
{

class O1DnSimulatorVulkan : public SimulatorVulkan<O1DnModel, O1DnModel_<VULKAN_DEVICE_LOCAL>>
{
private:
    Memory<O1DnModel_<VULKAN_DEVICE_LOCAL>, RAM> sensorMem_half_ram;

public:
    O1DnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<O1DnModel, O1DnModel_<VULKAN_DEVICE_LOCAL>>(map), sensorMem_half_ram(1) {}

    ~O1DnSimulatorVulkan() {}

    O1DnSimulatorVulkan(const O1DnSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(Memory<O1DnModel, RAM>& sensorMem);
};

using O1DnSimulatorVulkanPtr = std::shared_ptr<O1DnSimulatorVulkan>;

} // namespace rmagine
