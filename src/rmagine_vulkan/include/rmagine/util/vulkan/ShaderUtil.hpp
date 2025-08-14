#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <bit>

#define USE_GLSLANG_LIB
#if defined(USE_GLSLANG_LIB)
    #include <glslang/Include/glslang_c_interface.h>
    #include <glslang/Public/resource_limits_c.h>

    #include "rmagine/shaders/VulkanIncludeShader.hpp"
    #include "rmagine/shaders/VulkanShaderRGen.hpp"
    #include "rmagine/shaders/VulkanShaderCHit.hpp"
    #include "rmagine/shaders/VulkanShaderMiss.hpp"
#endif



namespace rmagine
{

enum ShaderType
{
    RGen,
    CHit,
    Miss,
    Call,
    //Last Element: only use it to get the size of this enum excluding this element
    SIZE
};

enum ShaderDefines
{
    //Sensor Defines
    Def_Sphere = 1<<0,
    Def_Pinhole = 1<<1,
    Def_O1Dn = 1<<2,
    Def_OnDn = 1<<3,
    //Result Defines
    Def_Hits = 1<<4,
    Def_Ranges = 1<<5,
    Def_Points = 1<<6,
    Def_Normals = 1<<7,
    Def_PrimitiveID = 1<<8,
    Def_GeometryID = 1<<9,
    Def_InstanceID = 1<<10,
    //Last Element: do not use
    END = 1<<11
};

//Bitmask for ShaderDefines
typedef uint32_t ShaderDefineFlags;



#if defined(USE_GLSLANG_LIB)
    glslang_stage_t get_glslang_stage(ShaderType shaderType);

    std::string get_shader_define_statements(ShaderDefineFlags shaderDefines);

    std::string get_shader_code(ShaderType shaderType, ShaderDefineFlags shaderDefines);
#endif

ShaderDefineFlags get_sensor_mask();

ShaderDefineFlags get_sensor_defines_from_flags(ShaderDefineFlags shaderDefines);

ShaderDefineFlags get_result_mask();

ShaderDefineFlags get_result_defines_from_flags(ShaderDefineFlags shaderDefines);

bool one_sensor_defined(ShaderDefineFlags shaderDefines);

std::string get_shader_info(ShaderType shaderType, ShaderDefineFlags shaderDefines);

/**
 * get a vector containing all the defines given the ShaderDefineFlags
 * 
 * @return std::vector containing the defines
 */
std::vector<std::string> get_shader_defines(ShaderDefineFlags shaderDefines);


} // namespace rmagine