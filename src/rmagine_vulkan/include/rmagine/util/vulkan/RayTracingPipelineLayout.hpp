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

class RayTracingPipelineLayout
{
private:
    DevicePtr device = nullptr;
    DescriptorSetLayoutPtr descriptorSetLayout = nullptr;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

public:
    RayTracingPipelineLayout(DevicePtr device, DescriptorSetLayoutPtr descriptorSetLayout);

    ~RayTracingPipelineLayout();

    RayTracingPipelineLayout(const RayTracingPipelineLayout&) = delete;


    VkPipelineLayout getPipelineLayout();

private:
    void createPipelineLayout();
};

using PipelineLayoutPtr = std::shared_ptr<RayTracingPipelineLayout>;

} // namespace rmagine
