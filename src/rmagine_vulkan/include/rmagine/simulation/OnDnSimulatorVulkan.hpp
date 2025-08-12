#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "SimulatorVulkan.hpp"



namespace rmagine
{

class OnDnSimulatorVulkan : public SimulatorVulkan<OnDnModel, OnDnModel_<VULKAN_DEVICE_LOCAL>>
{
private:
    Memory<OnDnModel_<VULKAN_DEVICE_LOCAL>, RAM> sensorMem_half_ram;

public:
    OnDnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<OnDnModel, OnDnModel_<VULKAN_DEVICE_LOCAL>>(map), sensorMem_half_ram(1) {}

    ~OnDnSimulatorVulkan() {}

    OnDnSimulatorVulkan(const OnDnSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(Memory<OnDnModel, RAM>& sensorMem_ram);
};

using OnDnSimulatorVulkanPtr = std::shared_ptr<OnDnSimulatorVulkan>;

} // namespace rmagine
