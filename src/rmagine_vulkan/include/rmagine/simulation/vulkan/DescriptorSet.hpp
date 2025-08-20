#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/vulkan/Device.hpp>
#include <rmagine/util/vulkan/DescriptorSetLayout.hpp>
#include <rmagine/util/vulkan/memory/Buffer.hpp>



namespace rmagine
{

//forward declaration
class AccelerationStructure;
using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;



class DescriptorSet
{
private:
    DevicePtr device = nullptr;
    DescriptorSetLayoutPtr descriptorSetLayout = nullptr;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

public:
    DescriptorSet();

    DescriptorSet(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout);

    ~DescriptorSet() {}

    DescriptorSet(const DescriptorSet&) = delete;

    void cleanup();

    void updateDescriptorSet(BufferPtr vertexBuffer, BufferPtr indexBuffer, 
                             BufferPtr sensorBuffer, BufferPtr resultsBuffer, 
                             BufferPtr tsbBuffer, BufferPtr tbmBuffer, 
                             AccelerationStructurePtr accelerationStructure);

    VkDescriptorSet* getDescriptorSetPtr();

private:
    void allocateDescriptorSet();
};

using DescriptorSetPtr = std::shared_ptr<DescriptorSet>;

} // namespace rmagine
