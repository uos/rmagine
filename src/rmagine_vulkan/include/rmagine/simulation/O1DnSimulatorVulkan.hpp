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
    Memory<Vector, VULKAN_DEVICE_LOCAL> dirs;

public:
    O1DnSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<O1DnModel>(map) {}

    ~O1DnSimulatorVulkan() {}

    O1DnSimulatorVulkan(const O1DnSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(const Memory<O1DnModel, RAM>& sensorMem);

    void updateAddresses(Memory<Transform, VULKAN_DEVICE_LOCAL>& tbmMem, Memory<VulkanResultsData, RAM>& resultsMem_ram);
};

using O1DnSimulatorVulkanPtr = std::shared_ptr<O1DnSimulatorVulkan>;

} // namespace rmagine
