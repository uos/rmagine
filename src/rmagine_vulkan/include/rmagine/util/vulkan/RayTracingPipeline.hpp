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
#include "ShaderUtil.hpp"



namespace rmagine
{

class RayTracingPipeline
{
private:
    DeviceWPtr device;
    RayTracingPipelineLayoutWPtr pipelineLayout;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

public:
    RayTracingPipeline(DeviceWPtr device, RayTracingPipelineLayoutWPtr pipelineLayout, ShaderDefineFlags shaderDefines);

    ~RayTracingPipeline();

    RayTracingPipeline(const RayTracingPipeline&) = delete;


    VkPipeline getPipeline();

private:
    void createPipelineCache();

    void createPipeline(ShaderDefineFlags shaderDefines);
};

using RayTracingPipelinePtr = std::shared_ptr<RayTracingPipeline>;

} // namespace rmagine
