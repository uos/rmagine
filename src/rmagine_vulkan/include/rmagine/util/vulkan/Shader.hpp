#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
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
    DevicePtr device = nullptr;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    
public:
    Shader(DevicePtr device, std::string path);

    Shader(DevicePtr device, ShaderType shaderType, ShaderDefineFlags shaderDefines);

    ~Shader() {}

    Shader(const Shader&) = delete;


    VkShaderModule getShaderModule();

    void cleanup();

private:
    void compileShader(ShaderType shaderType, ShaderDefineFlags shaderDefines, std::string sourcePath, std::string outputPath);

    void createShader(std::string path);
};

using ShaderPtr = std::shared_ptr<Shader>;

} // namespace rmagine
