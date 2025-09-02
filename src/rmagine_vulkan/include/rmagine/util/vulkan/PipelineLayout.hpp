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

class PipelineLayout
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

public:
    PipelineLayout(VulkanContextWPtr vulkan_context);

    ~PipelineLayout();

    PipelineLayout(const PipelineLayout&) = delete;


    VkPipelineLayout getPipelineLayout();

private:
    void createPipelineLayout();
};

using PipelineLayoutPtr = std::shared_ptr<PipelineLayout>;

} // namespace rmagine
