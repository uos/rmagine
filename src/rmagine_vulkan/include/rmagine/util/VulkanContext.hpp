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

#include "VulkanUtil.hpp"
#include "contextComponents/Device.hpp"
#include "contextComponents/CommandPool.hpp"
#include "contextComponents/DescriptorSetLayout.hpp"
#include "contextComponents/PipelineLayout.hpp"
#include "contextComponents/Pipeline.hpp"
#include "contextComponents/Shader.hpp"
#include "generalComponents/CommandBuffer.hpp"



namespace rmagine
{

class VulkanContext : std::enable_shared_from_this<VulkanContext>
{
private:
    ExtensionFunctionsPtr extensionFunctionsPtr = nullptr;

    DevicePtr device = nullptr;
    CommandPoolPtr commandPool = nullptr;
    DescriptorSetLayoutPtr descriptorSetLayout = nullptr;
    PipelineLayoutPtr pipelineLayout = nullptr;

    std::map<ShaderDefineFlags, PipelinePtr> pipelineMap = std::map<ShaderDefineFlags, PipelinePtr>();

    CommandBufferPtr defaultCommandBuffer = nullptr;

    std::map<ShaderDefineFlags, ShaderPtr> shaderMaps[ShaderType::SIZE] = {
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::RGen
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::CHit
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::Miss
        std::map<ShaderDefineFlags, ShaderPtr>()};    // ShaderType::Call

public:
    VulkanContext() : extensionFunctionsPtr(new ExtensionFunctions)
    {
        std::cout << "creating VulkanContext" << std::endl;

        device = std::make_shared<Device>();
        loadExtensionFunctions();

        commandPool = std::make_shared<CommandPool>(device);
        descriptorSetLayout = std::make_shared<DescriptorSetLayout>(device);
        pipelineLayout = std::make_shared<PipelineLayout>(device, descriptorSetLayout);

        defaultCommandBuffer = std::make_shared<CommandBuffer>(device, extensionFunctionsPtr, commandPool, pipelineLayout);

        std::cout << "VulkanContext created" << std::endl;
    }

    ~VulkanContext()
    {
        std::cout << "destroying VulkanContext" << std::endl;
        cleanup();
    }

    VulkanContext(const VulkanContext&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    PipelinePtr getPipeline(ShaderDefineFlags shaderDefines);

    ShaderPtr getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);

private:
    /**
     * functions from extensions are not available directly and have to be loaded manually via vkGetDeviceProcAddr()
     */
    void loadExtensionFunctions();

    /**
     * call cleanup functions of components
     */
    void cleanup();

public:
    void clearShaderCache();

    void clearPipelineCache();

    DevicePtr getDevice();

    CommandPoolPtr getCommandPool();

    DescriptorSetLayoutPtr getDescriptorSetLayout();

    PipelineLayoutPtr getPipelineLayout();

    ExtensionFunctionsPtr getExtensionFunctionsPtr();

    CommandBufferPtr getDefaultCommandBuffer();
};





using VulkanContextPtr = std::shared_ptr<VulkanContext>;
using VulkanContextWeakPtr = std::weak_ptr<VulkanContext>;

VulkanContextPtr get_vulkan_context();

VulkanContextWeakPtr get_vulkan_context_weak();

} // namespace rmagine
