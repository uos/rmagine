#include "rmagine/util/vulkan/Shader.hpp"



namespace rmagine
{

Shader::Shader(DevicePtr device, std::string path) : device(device)
{
    createShader(path);
}

Shader::Shader(DevicePtr device, ShaderType shaderType, ShaderDefineFlags shaderDefines) : device(device)
{
    std::string outputPath = get_program_dir() + get_shader_spv_path(shaderType, shaderDefines);

    bool forceShaderRecompile = false;
    #if defined(FORCE_SHADER_RECOMPILE)
        forceShaderRecompile = true;
    #endif  
    if(!std::filesystem::exists(outputPath) || forceShaderRecompile)
    {
        std::string sourcePath = get_program_dir() + get_shader_source_path(shaderType);
        compileShader(shaderType, shaderDefines, sourcePath, outputPath);
    }
    createShader(outputPath);
}



void Shader::createShader(std::string shaderPath)
{
    if(!std::filesystem::exists(shaderPath))
    {
        throw std::runtime_error("Shader binary file not found: " + shaderPath);
    }
    std::vector<char> shaderContents;
    if(std::ifstream shaderFile{shaderPath.c_str(), std::ios::binary | std::ios::ate})
    {
        const size_t fileSize = shaderFile.tellg();
        shaderFile.seekg(0);
        shaderContents.resize(fileSize, '\0');
        shaderFile.read(shaderContents.data(), fileSize);
        shaderFile.close();
    }

    if(shaderContents.size() == 0)
    {
        throw std::runtime_error("Shader binary file is empty: " + shaderPath);
    }

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = shaderContents.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderContents.data());
    
    if(vkCreateShaderModule(device->getLogicalDevice(), &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module: " + shaderPath);
    }
}

void Shader::compileShader(ShaderType shaderType, ShaderDefineFlags shaderDefines, std::string sourcePath, std::string outputPath)
{
    #if defined(USE_GLSLANG_LIB)
        //source:   https://github.com/KhronosGroup/glslang
        //see also: https://stackoverflow.com/questions/38234986/how-to-use-glslang
        //it seems that glslang sadly does not have a proper documentation

        //there is also a shaderc library - maybe use that one instead, if it works with raytracing pipeline shaders - maybe its better.

        std::string shaderCode = get_shader_code(shaderType, shaderDefines);
        
        glslang_initialize_process();

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
            printf("GLSL preprocessing failed %s\n", sourcePath.data());
            printf("%s\n", glslang_shader_get_info_log(shader));
            printf("%s\n", glslang_shader_get_info_debug_log(shader));
            printf("%s\n", input.code);
            glslang_shader_delete(shader);

            throw std::runtime_error("Failed compiling shader at: glslang_shader_preprocess!");
        }

        if (!glslang_shader_parse(shader, &input))
        {
            printf("GLSL parsing failed %s\n", sourcePath.data());
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
            printf("GLSL linking failed %s\n", sourcePath.data());
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
    #else
        std::vector<std::string> defines = get_shader_defines(shaderDefines);

        (void) shaderType;

        std::string command =  "/bin/glslangValidator --target-env vulkan1.3 ";
        std::string commandDefineMacro = "--define-macro ";
        std::string commandEqDef ="=def ";
        for(size_t i = 0; i < defines.size(); i++)
        {
            command = command + commandDefineMacro + defines[i] + commandEqDef;
        }
        command = command + "-o " + outputPath + " " + sourcePath;

        // a full command could look like this:
        // "/bin/glslangValidator --target-env vulkan1.3 "--define-macro DEFINE=def -o shaders/shaderName/DEFINE.spv shaderSources/shaderName.glsl"
        std::cout << "Compiling shader with the command: "  << command << std::endl;
        std::system(command.c_str());
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
