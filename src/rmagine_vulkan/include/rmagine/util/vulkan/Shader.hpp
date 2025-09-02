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

class Shader
{
private:
    VulkanContextWPtr vulkan_context;
    DevicePtr device = nullptr;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    
public:
    Shader(VulkanContextWPtr vulkan_context, ShaderType shaderType, ShaderDefineFlags shaderDefines);

    ~Shader();

    Shader(const Shader&) = delete;


    VkShaderModule getShaderModule();

private:
    std::vector<uint32_t> compileShader(ShaderType shaderType, ShaderDefineFlags shaderDefines);

    void createShader(std::vector<uint32_t> words);
};

using ShaderPtr = std::shared_ptr<Shader>;

} // namespace rmagine
