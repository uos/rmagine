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

    VkPipeline pipeline = VK_NULL_HANDLE;

    ShaderPtr rGenShader = nullptr;
    ShaderPtr cHitShader = nullptr;
    ShaderPtr missShader = nullptr;
    ShaderPtr callShader = nullptr;

public:
    RayTracingPipeline(DeviceWPtr device, RayTracingPipelineLayoutWPtr pipelineLayout, ShaderDefineFlags shaderDefines);

    ~RayTracingPipeline();

    RayTracingPipeline(const RayTracingPipeline&) = delete;


    VkPipeline getPipeline();

private:
    void createPipeline(ShaderDefineFlags shaderDefines);
};

using RayTracingPipelinePtr = std::shared_ptr<RayTracingPipeline>;

} // namespace rmagine
