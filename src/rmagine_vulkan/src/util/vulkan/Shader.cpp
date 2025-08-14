#include "rmagine/util/vulkan/Shader.hpp"



namespace rmagine
{

Shader::Shader(DevicePtr device, ShaderType shaderType, ShaderDefineFlags shaderDefines) : device(device)
{
    std::cout << "compiling & creating " << get_shader_info(shaderType, shaderDefines) << std::endl;
    createShader(compileShader(shaderType, shaderDefines));
}



void Shader::createShader(std::vector<uint32_t> words)
{
    if(words.size() == 0)
    {
        throw std::runtime_error("Shader binary is empty");
    }

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = words.size() * sizeof(uint32_t);
    shaderModuleCreateInfo.pCode = words.data();
    
    if(vkCreateShaderModule(device->getLogicalDevice(), &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module");
    }
}


std::vector<uint32_t> Shader::compileShader(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    #if defined(USE_GLSLANG_LIB)
        //source: https://github.com/KhronosGroup/glslang

        glslang_initialize_process();

        std::string shaderCode = get_shader_code(shaderType, shaderDefines);

        glslang_stage_t stage = get_glslang_stage(shaderType);

        glslang_input_t input{};
        input.language = GLSLANG_SOURCE_GLSL;
        input.stage = stage;
        input.client = GLSLANG_CLIENT_VULKAN;
        input.client_version = GLSLANG_TARGET_VULKAN_1_3;
        input.target_language = GLSLANG_TARGET_SPV;
        input.target_language_version = GLSLANG_TARGET_SPV_1_6;
        input.code = shaderCode.data();
        input.default_version = 460;
        input.default_profile = GLSLANG_NO_PROFILE; // GLSLANG_CORE_PROFILE;
        input.force_default_version_and_profile = false;
        input.forward_compatible = false;
        input.messages = GLSLANG_MSG_DEFAULT_BIT;
        input.resource = glslang_default_resource();

        glslang_shader_t* shader = glslang_shader_create(&input);

        if (!glslang_shader_preprocess(shader, &input))
        {
            printf("GLSL preprocessing failed %s\n", get_shader_info(shaderType, shaderDefines).data());
            printf("%s\n", glslang_shader_get_info_log(shader));
            printf("%s\n", glslang_shader_get_info_debug_log(shader));
            printf("%s\n", input.code);
            glslang_shader_delete(shader);

            throw std::runtime_error("Failed compiling shader at: glslang_shader_preprocess!");
        }

        if (!glslang_shader_parse(shader, &input))
        {
            printf("GLSL parsing failed %s\n", get_shader_info(shaderType, shaderDefines).data());
            printf("%s\n", glslang_shader_get_info_log(shader));
            printf("%s\n", glslang_shader_get_info_debug_log(shader));
            printf("%s\n", glslang_shader_get_preprocessed_code(shader));
            glslang_shader_delete(shader);

            throw std::runtime_error("Failed compiling shader at: glslang_shader_parse!");
        }

        glslang_program_t* program = glslang_program_create();
        glslang_program_add_shader(program, shader);
        
        if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
        {
            printf("GLSL linking failed %s\n", get_shader_info(shaderType, shaderDefines).data());
            printf("%s\n", glslang_program_get_info_log(program));
            printf("%s\n", glslang_program_get_info_debug_log(program));
            glslang_program_delete(program);
            glslang_shader_delete(shader);

            throw std::runtime_error("Failed compiling shader at: glslang_program_link!");
        }

        glslang_program_SPIRV_generate(program, stage);

        size_t size = glslang_program_SPIRV_get_size(program);
        std::vector<uint32_t> words(size);
        glslang_program_SPIRV_get(program, words.data());

        glslang_program_delete(program);
        glslang_shader_delete(shader);

        glslang_finalize_process();

        return words;
    #endif
}


VkShaderModule Shader::getShaderModule()
{
    return shaderModule;
}


void Shader::cleanup()
{
    if(shaderModule != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device->getLogicalDevice(), shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

} // namespace rmagine
