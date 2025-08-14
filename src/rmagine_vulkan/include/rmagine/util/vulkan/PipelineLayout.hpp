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
#include "DescriptorSetLayout.hpp"



namespace rmagine
{

class PipelineLayout
{
private:
    DevicePtr device = nullptr;
    DescriptorSetLayoutPtr descriptorSetLayout = nullptr;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

public:
    PipelineLayout(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout) : device(device), descriptorSetLayout(descriptorSetLayout)
    {
        createPipelineLayout();
    }

    ~PipelineLayout() {}

    PipelineLayout(const PipelineLayout&) = delete;


    void cleanup();

    VkPipelineLayout getPipelineLayout();

private:
    void createPipelineLayout();
};

using PipelineLayoutPtr = std::shared_ptr<PipelineLayout>;

} // namespace rmagine
