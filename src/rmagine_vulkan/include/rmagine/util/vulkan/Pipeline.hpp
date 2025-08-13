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

#include <rmagine/util/VulkanUtil.hpp>
#include "Device.hpp"
#include "PipelineLayout.hpp"
#include "ShaderUtil.hpp"



namespace rmagine
{

//forward declaration
class ShaderBindingTable;
using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;



class Pipeline : public std::enable_shared_from_this<Pipeline>
{
private:
    DevicePtr device = nullptr;
    PipelineLayoutPtr pipelineLayout = nullptr;
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    ShaderBindingTablePtr shaderBindingTable = nullptr;

public:
    Pipeline(ShaderDefineFlags shaderDefines);

    Pipeline(DevicePtr device, PipelineLayoutPtr pipelineLayout, ExtensionFunctionsPtr extensionFunctionsPtr, ShaderDefineFlags shaderDefines);

    ~Pipeline() {}

    Pipeline(const Pipeline&) = delete;


    void createShaderBindingTable();

    void cleanup();

    VkPipeline getPipeline();

    ShaderBindingTablePtr getShaderBindingTable();

private:
    void createPipelineCache();

    void createPipeline(ShaderDefineFlags shaderDefines);
};

using PipelinePtr = std::shared_ptr<Pipeline>;

} // namespace rmagine
