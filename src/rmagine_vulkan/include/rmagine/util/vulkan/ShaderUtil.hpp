#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <bit>

// #define USE_GLSLANG_LIB
#if defined(USE_GLSLANG_LIB)
    #include <glslang/Include/glslang_c_interface.h>
    #include <glslang/Public/resource_limits_c.h>
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
#endif

ShaderDefineFlags get_sensor_mask();

ShaderDefineFlags get_result_mask();

bool one_sensor_defined(ShaderDefineFlags shaderDefines);



/**
 * returns absolute path to the bin directory
 */
std::string get_program_dir();



/**
 * returns path to shadersource given the arguments
 * 
 * @return returned path should look like this: shader_sources_dir/shaderType.fileEnding
 */
std::string get_shader_source_path(ShaderType shaderType);

/**
 * returns path to .spv file given the arguments
 * 
 * @return returned path should look like this: shaders_spv_dir/shaderType/DEFINE1_DEFINE2_DEFINE3.fileEnding
 */
std::string get_shader_spv_path(ShaderType shaderType, ShaderDefineFlags shaderDefines);

/**
 * get a vector containing all the defines given the ShaderDefineFlags
 * 
 * @return std::vector containing the defines
 */
std::vector<std::string> get_shader_defines(ShaderDefineFlags shaderDefines);


} // namespace rmagine