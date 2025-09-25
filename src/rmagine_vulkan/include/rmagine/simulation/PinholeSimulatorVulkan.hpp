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

class PinholeSimulatorVulkan : public SimulatorVulkan<PinholeModel>
{
public:
    PinholeSimulatorVulkan() : SimulatorVulkan<PinholeModel>() {}

    PinholeSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<PinholeModel>(map) {}

    ~PinholeSimulatorVulkan() {}

    PinholeSimulatorVulkan(const PinholeSimulatorVulkan& other) : SimulatorVulkan<PinholeModel>(other) {}


    void setModel(const Memory<PinholeModel, RAM>& sensorMem_ram);
    void setModel(const PinholeModel& sensor);
};

using PinholeSimulatorVulkanPtr = std::shared_ptr<PinholeSimulatorVulkan>;

} // namespace rmagine
