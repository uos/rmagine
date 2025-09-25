#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanUtil.hpp>
#include "Device.hpp"



namespace rmagine
{

class DescriptorSetLayout
{
private:
    DevicePtr device = nullptr;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

public:
    DescriptorSetLayout(DevicePtr device);

    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;


    VkDescriptorPool getDescriptorPool();
    
    VkDescriptorSetLayout* getDescriptorSetLayoutPtr();

private:
    void createDescriptorPool();

    void createDescriptorSetLayout();
};

using DescriptorSetLayoutPtr = std::shared_ptr<DescriptorSetLayout>;

} // namespace rmagine
