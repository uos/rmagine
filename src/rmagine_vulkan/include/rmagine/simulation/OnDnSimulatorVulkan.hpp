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
    Memory<Vector, DEVICE_LOCAL_VULKAN> origs;
    Memory<Vector, DEVICE_LOCAL_VULKAN> dirs;

public:
    OnDnSimulatorVulkan() : SimulatorVulkan<OnDnModel>() {}

    OnDnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<OnDnModel>(map) {}

    ~OnDnSimulatorVulkan() {}

    OnDnSimulatorVulkan(const OnDnSimulatorVulkan& other) : SimulatorVulkan<OnDnModel>(other)
    {
        origs.resize(other.origs.size());
        origs = other.origs;

        dirs.resize(other.dirs.size());
        dirs = other.dirs;
    }


    void setModel(const Memory<OnDnModel, RAM>& sensorMem_ram);
    void setModel(const OnDnModel& sensor);

    void updateTbmAndSensorSpecificAddresses(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem);
};

using OnDnSimulatorVulkanPtr = std::shared_ptr<OnDnSimulatorVulkan>;

} // namespace rmagine
