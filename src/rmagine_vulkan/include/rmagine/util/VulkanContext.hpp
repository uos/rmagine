#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>

#include <vulkan/vulkan.h>

#include "VulkanUtil.hpp"
#include "vulkan/Device.hpp"
#include "vulkan/CommandPool.hpp"
#include "vulkan/DescriptorSetLayout.hpp"
#include "vulkan/RayTracingPipelineLayout.hpp"
#include "vulkan/Shader.hpp"



namespace rmagine
{

//forward declaration
class ShaderBindingTable;
using ShaderBindingTablePtr = std::shared_ptr<ShaderBindingTable>;



class VulkanContext : public std::enable_shared_from_this<VulkanContext>
{
private:
    DevicePtr device = nullptr;

    CommandPoolPtr commandPool = nullptr;
    DescriptorSetLayoutPtr descriptorSetLayout = nullptr;
    PipelineLayoutPtr pipelineLayout = nullptr;

    std::map<ShaderDefineFlags, ShaderPtr> shaderMaps[ShaderType::SHADER_TYPE_SIZE] = {
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::RGen
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::CHit
        std::map<ShaderDefineFlags, ShaderPtr>(),     // ShaderType::Miss
        std::map<ShaderDefineFlags, ShaderPtr>()};    // ShaderType::Call

    std::map<ShaderDefineFlags, ShaderBindingTablePtr> shaderBindingTableMap = std::map<ShaderDefineFlags, ShaderBindingTablePtr>();

    // two different threads trying to access the shader-maps/sbt-map at the same time might cause issues without these mutexes
    std::mutex shaderMutex;
    std::mutex sbtMutex;

public:
    ExtensionFunctions extensionFuncs;


    VulkanContext();

    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;//delete copy connstructor, you should never need to copy an instance of this class, and doing so may cause issues


    ShaderPtr getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);
    void removeShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);
    size_t getShaderCacheSize();
    void clearShaderCache();

    ShaderBindingTablePtr getShaderBindingTable(ShaderDefineFlags shaderDefines);
    void removeShaderBindingTable(ShaderDefineFlags shaderDefines);
    size_t getShaderBindingTableCacheSize();
    void clearShaderBindingTableCache();

    DevicePtr getDevice();
    CommandPoolPtr getCommandPool();
    DescriptorSetLayoutPtr getDescriptorSetLayout();
    PipelineLayoutPtr getPipelineLayout();

private:
    /**
     * functions from extensions are not available directly and have to be loaded manually via vkGetDeviceProcAddr()
     */
    void loadExtensionFunctions();
};

using VulkanContextPtr = std::shared_ptr<VulkanContext>;
using VulkanContextWPtr = std::weak_ptr<VulkanContext>;



VulkanContextPtr get_vulkan_context();

VulkanContextWPtr get_vulkan_context_weak();

} // namespace rmagine
