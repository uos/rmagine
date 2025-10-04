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
    VulkanContextPtr vulkan_context = nullptr;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

public:
    DescriptorSet(VulkanContextPtr vulkan_context);

    ~DescriptorSet();

    DescriptorSet(const DescriptorSet&) = delete;


    void updateDescriptorSet(AccelerationStructurePtr accelerationStructure, BufferPtr mapDataBuffer, 
                             BufferPtr sensorBuffer, BufferPtr resultsBuffer, 
                             BufferPtr tsbBuffer, BufferPtr origsDirsAndTransformsBuffer);

    VkDescriptorSet* getDescriptorSetPtr();

private:
    /**
     * each DescriptorSet needs its own DescriptorPool, as they are externally synchronised and multithreading might cause issues otherwise.
     * see: https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPool.html
     */
    void createDescriptorPool();

    void allocateDescriptorSet();
};

using DescriptorSetPtr = std::shared_ptr<DescriptorSet>;

} // namespace rmagine
