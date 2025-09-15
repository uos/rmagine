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

class OnDnSimulatorVulkan : public SimulatorVulkan<OnDnModel>
{
private:
    Memory<Vector, VULKAN_DEVICE_LOCAL> origs;
    Memory<Vector, VULKAN_DEVICE_LOCAL> dirs;

public:
    OnDnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<OnDnModel>(map) {}

    ~OnDnSimulatorVulkan() {}

    OnDnSimulatorVulkan(const OnDnSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(const Memory<OnDnModel, RAM>& sensorMem_ram);
    void setModel(const OnDnModel& sensor);

    void updateTbmAndSensorSpecificAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem);
};

using OnDnSimulatorVulkanPtr = std::shared_ptr<OnDnSimulatorVulkan>;

} // namespace rmagine
