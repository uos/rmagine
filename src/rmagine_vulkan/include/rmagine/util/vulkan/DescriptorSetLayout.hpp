#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

#include <vulkan/vulkan.h>

#include <rmagine/util/VulkanContextUtil.hpp>
#include "Device.hpp"



namespace rmagine
{

class DescriptorSetLayout
{
private:
    DeviceWPtr device;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

public:
    DescriptorSetLayout(DeviceWPtr device);

    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;

    
    VkDescriptorSetLayout* getDescriptorSetLayoutPtr();

private:
    void createDescriptorSetLayout();
};

using DescriptorSetLayoutPtr = std::shared_ptr<DescriptorSetLayout>;

} // namespace rmagine
