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

class RayTracingPipelineLayout
{
private:
    DeviceWPtr device;
    DescriptorSetLayoutWPtr descriptorSetLayout;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

public:
    RayTracingPipelineLayout(DeviceWPtr device, DescriptorSetLayoutWPtr descriptorSetLayout);

    ~RayTracingPipelineLayout();

    RayTracingPipelineLayout(const RayTracingPipelineLayout&) = delete;


    VkPipelineLayout getPipelineLayout();

private:
    void createPipelineLayout();
};

using RayTracingPipelineLayoutPtr = std::shared_ptr<RayTracingPipelineLayout>;

} // namespace rmagine
