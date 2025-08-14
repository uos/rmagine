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

class PinholeSimulatorVulkan : public SimulatorVulkan<PinholeModel, PinholeModel>
{
public:
    PinholeSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<PinholeModel, PinholeModel>(map) {}

    ~PinholeSimulatorVulkan() {}

    PinholeSimulatorVulkan(const PinholeSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(Memory<PinholeModel, RAM>& sensorMem_ram);
};

using PinholeSimulatorVulkanPtr = std::shared_ptr<PinholeSimulatorVulkan>;

} // namespace rmagine
