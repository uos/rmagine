#include "ShaderUtil.hpp"



namespace rmagine
{

const std::string shader_sources_dir = "shaderSources";
const std::string shaders_spv_dir = "shaders";

const std::string get_shader_names[ShaderType::SIZE] = {
    "RayGeneration",
    "ClosestHit",
    "Miss",
    "Callable",
};
const std::string get_shader_endings[ShaderType::SIZE] = {
    ".rgen",
    ".rchit",
    ".rmiss",
    ".rcall",
};
#if defined(USE_GLSLANG_LIB)
    const glslang_stage_t get_glslang_stage_t[ShaderType::SIZE] = {
        GLSLANG_STAGE_RAYGEN,
        GLSLANG_STAGE_CLOSESTHIT,
        GLSLANG_STAGE_MISS,
        GLSLANG_STAGE_CALLABLE,
    };
#endif

const std::string shader_spv_ending = ".spv";

const std::map<ShaderDefines, std::string> get_shader_define = {
    //Sensor Defines
    {ShaderDefines::Def_Sphere,       "SPHERE"      },
    {ShaderDefines::Def_Pinhole,      "PINHOLE"     },
    {ShaderDefines::Def_O1Dn,         "O1DN"        },
    {ShaderDefines::Def_OnDn,         "ONDN"        },
    //Result Defines
    {ShaderDefines::Def_Hits,         "HITS"        },
    {ShaderDefines::Def_Ranges,       "RANGES"      },
    {ShaderDefines::Def_Points,       "POINTS"      },
    {ShaderDefines::Def_Normals,      "NORMALS"     },
    {ShaderDefines::Def_PrimitiveID,  "PRIMITIVE_ID"},
    {ShaderDefines::Def_GeometryID,   "GEOMETRY_ID" },
    {ShaderDefines::Def_InstanceID,   "INSTANCE_ID" },
    //You should not access this
    {ShaderDefines::END,              "ERROR"       }};



#if defined(USE_GLSLANG_LIB)
    glslang_stage_t get_glslang_stage(ShaderType shaderType)
    {
        return get_glslang_stage_t[shaderType];
    }
#endif

ShaderDefineFlags get_sensor_mask()
{
    return ShaderDefines::Def_Sphere | ShaderDefines::Def_Pinhole | ShaderDefines::Def_O1Dn | ShaderDefines::Def_OnDn;
}

ShaderDefineFlags get_result_mask()
{
    return ShaderDefines::Def_Hits | ShaderDefines::Def_Ranges | ShaderDefines::Def_Points | ShaderDefines::Def_Normals | ShaderDefines::Def_PrimitiveID | ShaderDefines::Def_GeometryID | ShaderDefines::Def_InstanceID;
}

bool one_sensor_defined(ShaderDefineFlags shaderDefines)
{
    ShaderDefineFlags maskedShaderDefines = shaderDefines & get_sensor_mask();
    return std::has_single_bit(maskedShaderDefines);
}



std::string program_dir = "";

std::string get_program_dir()
{
    if(program_dir == "")
    {
        #if defined(__linux__) || defined(linux) || defined(__linux)
            program_dir = std::filesystem::canonical("/proc/self/exe").remove_filename().c_str();
        #elif defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__) || defined(WIN32) || defined(__WIN32__) || defined(__NT__)
            //TODO: For Windows: GetModuleFileNameA(NULL)
            throw std::runtime_error("get_program_dir() not done for windows!");
        #else
            throw std::runtime_error("get_program_dir() not done for whatever operating sytem you are using!");
        #endif
    }
    return program_dir;
}



std::string get_shader_source_path(ShaderType shaderType)
{
    return shader_sources_dir + "/" + get_shader_names[shaderType] + get_shader_endings[shaderType];
}

std::string get_shader_spv_path(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::END)
    {
        throw std::invalid_argument("invalid shader defines!");
    }

    std::string path = shaders_spv_dir + "/" + get_shader_names[shaderType] + "/";

    std::vector<std::string> defines = get_shader_defines(shaderDefines);
    for(size_t i = 0; i < defines.size(); i++)
    {
        path = path + (i != 0 ? "_" : "") + defines[i];
    }

    return path + shader_spv_ending;
}

std::vector<std::string> get_shader_defines(ShaderDefineFlags shaderDefines)
{
    std::vector<std::string> defines = std::vector<std::string>();
    for(ShaderDefineFlags i = 1; i < ShaderDefines::END; i = i<<1)
    {
        if(i & shaderDefines)
        {
            defines.push_back(get_shader_define.at((ShaderDefines)i));
        }
    }
    return defines;
}

} // namespace rmagine