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
class TopLevelAccelerationStructure;
using TopLevelAccelerationStructurePtr = std::shared_ptr<TopLevelAccelerationStructure>;



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
                             TopLevelAccelerationStructurePtr topLevelAccelerationStructure);

    VkDescriptorSet* getDescriptorSetPtr();

private:
    void allocateDescriptorSet();
};

using DescriptorSetPtr = std::shared_ptr<DescriptorSet>;

} // namespace rmagine
