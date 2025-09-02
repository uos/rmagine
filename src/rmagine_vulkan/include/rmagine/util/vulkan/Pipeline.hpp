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
#include "ShaderUtil.hpp"



namespace rmagine
{

class Pipeline
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

public:
    Pipeline(VulkanContextWPtr vulkan_context, ShaderDefineFlags shaderDefines);

    ~Pipeline();

    Pipeline(const Pipeline&) = delete;


    VkPipeline getPipeline();

private:
    void createPipelineCache();

    void createPipeline(ShaderDefineFlags shaderDefines);
};

using PipelinePtr = std::shared_ptr<Pipeline>;

} // namespace rmagine
