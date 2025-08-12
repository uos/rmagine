#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include "Device.hpp"
#include "../VulkanUtil.hpp"



namespace rmagine
{

class DescriptorSetLayout
{
private:
    DevicePtr device = nullptr;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

public:
    DescriptorSetLayout(DevicePtr device) : device(device)
    {
        createDescriptorPool();
        createDescriptorSetLayout();
    }

    ~DescriptorSetLayout() {}

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;

    void cleanup();

    VkDescriptorPool getDescriptorPool();
    
    VkDescriptorSetLayout* getDescriptorSetLayoutPtr();

private:
    void createDescriptorPool();

    void createDescriptorSetLayout();
};

using DescriptorSetLayoutPtr = std::shared_ptr<DescriptorSetLayout>;

} // namespace rmagine
