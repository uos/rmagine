#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <bit>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include "rmagine/shaders/VulkanIncludeShader.hpp"
#include "rmagine/shaders/VulkanShaderRGen.hpp"
#include "rmagine/shaders/VulkanShaderCHit.hpp"
#include "rmagine/shaders/VulkanShaderMiss.hpp"



namespace rmagine
{

enum ShaderType
{
    RGen,
    CHit,
    Miss,
    Call,
    //Last Element: only use it to get the size of this enum excluding this element
    SHADER_TYPE_SIZE
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
    SHADER_DEFINES_END = 1<<11
};

//Bitmask for ShaderDefines
typedef uint32_t ShaderDefineFlags;



glslang_stage_t get_glslang_stage(ShaderType shaderType);

/**
 * get a vector containing all the defines given the ShaderDefineFlags
 * 
 * @param shaderDefines shader define flags
 * 
 * @return std::vector containing the defines
 */
std::vector<std::string> get_shader_defines(ShaderDefineFlags shaderDefines);

/**
 * get a string containing define statements: "#define DEFINE_A\n#define DEFINE_B\n"
 * 
 * @param shaderDefines shader define flags
 * 
 * @return string containing define statements
 */
std::string get_shader_define_statements(ShaderDefineFlags shaderDefines);

/**
 * get a string giving some information about the shader described by the shaderType & shaderDefines for debugging/loging
 * 
 * @param shaderType shader type
 * 
 * @param shaderDefines shader define flags
 * 
 * @return string containing shader information
 */
std::string get_shader_info(ShaderType shaderType, ShaderDefineFlags shaderDefines);

/**
 * get a string giving some information about the shader defines described by the shaderDefines for debugging/loging
 * 
 * @param shaderDefines shader define flags
 * 
 * @return string containing shader defines information
 */
std::string get_shader_defines_info(ShaderDefineFlags shaderDefines);

/**
 * get a string containing the entire code for the shader described by the shaderType & shaderDefines
 * 
 * @param shaderType shader type
 * 
 * @param shaderDefines shader define flags
 * 
 * @return string containing shader code
 */
std::string get_shader_code(ShaderType shaderType, ShaderDefineFlags shaderDefines);


ShaderDefineFlags get_sensor_mask();

ShaderDefineFlags get_sensor_defines_from_flags(ShaderDefineFlags shaderDefines);

ShaderDefineFlags get_result_mask();

ShaderDefineFlags get_result_defines_from_flags(ShaderDefineFlags shaderDefines);


/**
 * check if the shaderDefines correctly define only one shader 
 * 
 * @param shaderDefines shader define flags
 * 
 * @return true if only one shader is defined in the shaderDefines
 */
bool one_sensor_defined(ShaderDefineFlags shaderDefines);



} // namespace rmagine