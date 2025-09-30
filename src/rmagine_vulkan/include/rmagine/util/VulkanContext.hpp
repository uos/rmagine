#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>

#include <vulkan/vulkan.h>

#include "VulkanContextUtil.hpp"
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
    RayTracingPipelineLayoutPtr pipelineLayout = nullptr;

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


    /**
     * returns a shader of the chosen type, compiled with chosen defines
     * not applicable defines get ignored
     * if the requested shader does not exist it gets created and stored in a map (cache)
     * to avoid having to recreate the shader should it get requested again
     * 
     * @param shaderType Type of the shader (Ray generation, Closest hit, Miss)
     * 
     * @param shaderDefines defines used for compilation
     * 
     * @return the described shader
     */
    ShaderPtr getShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);
    /**
     * remove the described shader from the shader cache
     * (shaders still in use only get deleted afterwards)
     * 
     * @param shaderType Type of the shader (Ray generation, Closest hit, Miss)
     * 
     * @param shaderDefines defines used for compilation
     */
    void removeShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);
    /**
     * get the number of shaders currently in the shader cache
     */
    size_t getShaderCacheSize();
    /**
     * remove all shaders from the shader cache
     * (shaders still in use only get deleted afterwards)
     */
    void clearShaderCache();

    /**
     * get the shader binding table (and ray tracing pipeline) corresponding to the chosen defines
     * if the requested shader binding table does not exist it gets created and stored in a map (cache)
     * to avoid having to recreate the shader should it get requested again
     * 
     * @param shaderDefines defines used for compilation of the shaders used by the ray tracing pipeline
     * 
     * @return the described shader binding table
     */
    ShaderBindingTablePtr getShaderBindingTable(ShaderDefineFlags shaderDefines);
    /**
     * remove the described shader binding table from the shader cache
     * (shaders still in use only get deleted afterwards)
     * 
     * @param shaderDefines defines used for compilation of the shaders used by the ray tracing pipeline
     */
    void removeShaderBindingTable(ShaderDefineFlags shaderDefines);
    /**
     * get the number of shader binding tables currently in the shader binding table cache
     */
    size_t getShaderBindingTableCacheSize();
    /**
     * remove all shader binding tables from the shader binding table cache
     * (shader binding tables still in use only get deleted afterwards)
     */
    void clearShaderBindingTableCache();

    DevicePtr getDevice();
    CommandPoolPtr getCommandPool();
    DescriptorSetLayoutPtr getDescriptorSetLayout();
    RayTracingPipelineLayoutPtr getPipelineLayout();

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
