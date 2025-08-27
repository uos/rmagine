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
    SphereSimulatorVulkan(VulkanMapPtr map) : SimulatorVulkan<SphericalModel>(map) {}

    ~SphereSimulatorVulkan() {}

    SphereSimulatorVulkan(const SphereSimulatorVulkan&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    void setModel(const Memory<SphericalModel, RAM>& sensorMem_ram);
};

using SphereSimulatorVulkanPtr = std::shared_ptr<SphereSimulatorVulkan>;

} // namespace rmagine
