#include "rmagine/util/vulkan/ShaderUtil.hpp"



namespace rmagine
{

const std::string get_shader_names[ShaderType::SHADER_TYPE_SIZE] = {
    "RayGeneration",
    "ClosestHit",
    "Miss",
    "Callable",
};

const glslang_stage_t get_glslang_stage_t[ShaderType::SHADER_TYPE_SIZE] = {
    GLSLANG_STAGE_RAYGEN,
    GLSLANG_STAGE_CLOSESTHIT,
    GLSLANG_STAGE_MISS,
    GLSLANG_STAGE_CALLABLE,
};

const std::map<ShaderDefines, std::string> get_shader_define = {
    //Sensor Defines
    {ShaderDefines::Def_Sphere,          "SPHERE"      },
    {ShaderDefines::Def_Pinhole,         "PINHOLE"     },
    {ShaderDefines::Def_O1Dn,            "O1DN"        },
    {ShaderDefines::Def_OnDn,            "ONDN"        },
    //Result Defines
    {ShaderDefines::Def_Hits,            "HITS"        },
    {ShaderDefines::Def_Ranges,          "RANGES"      },
    {ShaderDefines::Def_Points,          "POINTS"      },
    {ShaderDefines::Def_Normals,         "NORMALS"     },
    {ShaderDefines::Def_PrimitiveID,     "PRIMITIVE_ID"},
    {ShaderDefines::Def_GeometryID,      "GEOMETRY_ID" },
    {ShaderDefines::Def_InstanceID,      "INSTANCE_ID" },
    //You should not access this
    {ShaderDefines::SHADER_DEFINES_END, "ERROR"       }};



glslang_stage_t get_glslang_stage(ShaderType shaderType)
{
    return get_glslang_stage_t[shaderType];
}

std::string get_shader_define_statements(ShaderDefineFlags shaderDefines)
{
    std::vector<std::string> defines = get_shader_defines(shaderDefines);

    std::string shaderCodeDefines = "";
    for(size_t i = 0; i < defines.size(); i++)
    {
        shaderCodeDefines += "#define " + defines[i] + "\n";
    }

    return shaderCodeDefines;
}

std::string get_shader_code(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    std::string shaderCode = "";

    switch (shaderType)
    {
    case ShaderType::RGen:
        shaderCode += rgen_preamble;
        break;
    case ShaderType::CHit:
        shaderCode += chit_preamble;
        break;
    case ShaderType::Miss:
        shaderCode += miss_preamble;
        break;
    default:
        throw std::invalid_argument("illegal ShaderType");
        break;
    }

    shaderCode += get_shader_define_statements(shaderDefines);

    shaderCode += util_code;
    
    switch (shaderType)
    {
    case ShaderType::RGen:
        shaderCode += rgen_code;
        break;
    case ShaderType::CHit:
        shaderCode += chit_code;
        break;
    case ShaderType::Miss:
        shaderCode += miss_code;
        break;
    default:
        throw std::invalid_argument("illegal ShaderType");
        break;
    }

    return shaderCode;
}

ShaderDefineFlags get_sensor_mask()
{
    return ShaderDefines::Def_Sphere | ShaderDefines::Def_Pinhole | ShaderDefines::Def_O1Dn | ShaderDefines::Def_OnDn;
}

ShaderDefineFlags get_sensor_defines_from_flags(ShaderDefineFlags shaderDefines)
{
    return shaderDefines & get_sensor_mask();
}

ShaderDefineFlags get_result_mask()
{
    return ShaderDefines::Def_Hits | ShaderDefines::Def_Ranges | ShaderDefines::Def_Points | ShaderDefines::Def_Normals | ShaderDefines::Def_PrimitiveID | ShaderDefines::Def_GeometryID | ShaderDefines::Def_InstanceID;
}

ShaderDefineFlags get_result_defines_from_flags(ShaderDefineFlags shaderDefines)
{
    return shaderDefines & get_result_mask();
}

bool one_sensor_defined(ShaderDefineFlags shaderDefines)
{
    ShaderDefineFlags maskedShaderDefines = shaderDefines & get_sensor_mask();
    // return std::has_single_bit(maskedShaderDefines); // works only from c++20 onwards
    return maskedShaderDefines && !(maskedShaderDefines & (maskedShaderDefines-1)); 
}

std::string get_shader_info(ShaderType shaderType, ShaderDefineFlags shaderDefines)
{
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
    {
        throw std::invalid_argument("invalid shader defines!");
    }
    if(shaderType >= ShaderType::SHADER_TYPE_SIZE)
    {
        throw std::invalid_argument("invalid shader type!");
    }

    std::string info = get_shader_names[shaderType] + ": ";

    std::vector<std::string> defines = get_shader_defines(shaderDefines);
    for(size_t i = 0; i < defines.size(); i++)
    {
        info += (i != 0 ? ", " : "") + defines[i];
    }

    return info;
}

std::vector<std::string> get_shader_defines(ShaderDefineFlags shaderDefines)
{
    if(shaderDefines == 0 || shaderDefines >= ShaderDefines::SHADER_DEFINES_END)
    {
        throw std::invalid_argument("invalid shader defines!");
    }

    std::vector<std::string> defines = std::vector<std::string>();
    for(ShaderDefineFlags i = 1; i < ShaderDefines::SHADER_DEFINES_END; i = i<<1)
    {
        if(i & shaderDefines)
        {
            defines.push_back(get_shader_define.at((ShaderDefines)i));
        }
    }

    return defines;
}

} // namespace rmagine