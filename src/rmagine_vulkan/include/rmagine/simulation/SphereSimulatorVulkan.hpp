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

class SphereSimulatorVulkan : public SimulatorVulkan<SphericalModel>
{
public:
    SphereSimulatorVulkan() : SimulatorVulkan<SphericalModel>() {}

    SphereSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<SphericalModel>(map) {}

    ~SphereSimulatorVulkan() {}

    SphereSimulatorVulkan(const SphereSimulatorVulkan& other) : SimulatorVulkan<SphericalModel>(other) {}


    void setModel(const Memory<SphericalModel, RAM>& sensorMem_ram);
    void setModel(const SphericalModel& sensor);
};

using SphereSimulatorVulkanPtr = std::shared_ptr<SphereSimulatorVulkan>;

} // namespace rmagine
