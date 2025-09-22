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

class O1DnSimulatorVulkan : public SimulatorVulkan<O1DnModel>
{
private:
    Memory<Vector, DEVICE_LOCAL_VULKAN> dirs;


public:
    O1DnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<O1DnModel>(map) {}

    ~O1DnSimulatorVulkan() {}

    O1DnSimulatorVulkan(const O1DnSimulatorVulkan& other) : SimulatorVulkan<O1DnModel>(other)
    {
        dirs.resize(other.dirs.size());
        dirs = other.dirs;
    }


    void setModel(const Memory<O1DnModel, RAM>& sensorMem_ram);
    void setModel(const O1DnModel& sensor);

    void updateTbmAndSensorSpecificAddresses(Memory<Transform, DEVICE_LOCAL_VULKAN>& tbmMem);
};

using O1DnSimulatorVulkanPtr = std::shared_ptr<O1DnSimulatorVulkan>;

} // namespace rmagine
