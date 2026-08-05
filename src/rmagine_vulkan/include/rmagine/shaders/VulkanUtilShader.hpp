#pragma once

#include <string>

//TODO: put strings in external shaderfiles and include them with @INCLUDE_STRING@. example in cmake/FileToString.h.in


/**
 * this shadercode contains structs, functions and layouts that may be usefull in multiple shaders
 * this code is added to the beginn ing of all other shaders (RGen, CHit, Miss)
 */
static const std::string util_code = R""""(#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_ARB_shading_language_include : require
#extension GL_EXT_shader_explicit_arithmetic_types : require



// the memory struct sent to the GPU contains a DataT* and a size_t, which are depending on the CPU 32 or 64 Bit long
#ifndef _32BIT
    #define _64BIT
#endif



layout(buffer_reference, std430, buffer_reference_align = 4) buffer float_array //cant be a vec3 array as that would introduce 4 bytes of padding 
{
    float f;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer uint_array //cant be a uvec3 array as that would introduce 4 bytes of padding 
{
    uint i;
};

layout(buffer_reference, std430, buffer_reference_align = 1) buffer uint8_array  
{
    uint8_t i;
};



struct Transform
{
    vec4 rot;
    vec3 pos;
    int stamp;
};

struct Interval
{
    float rayMin;
    float rayMax;
};

struct DiscreteInterval
{
    float angleMin;
    float angleInc;
    int size;
};

// this is just a placeholder-struct, 
// as nothing in this shader actually reads from this struct
// but it is part of the o1dn and ondn sensors
struct Memory
{
    #if defined(_64BIT)
        uint64_t ptr;
        uint64_t size;
    #elif defined(_32BIT)
        int ptr;
        int size;
    #endif
};

struct Result
{
    uint64_t hitsAddress;
    uint64_t rangesAddress;
    uint64_t pointsAddress;
    uint64_t normalsAddress;
    uint64_t primitiveIdAddress;
    uint64_t instanceIdAddress;
    uint64_t geometryIdAddress;
};



vec4 invQuaternion(vec4 q)
{
    return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 multQuaternions(vec4 q1, vec4 q2)
{
    return vec4(q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
                q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
                q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
                q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z);
}

vec3 rotateVec3(vec4 q, vec3 v)
{
    return multQuaternions(multQuaternions(q, vec4(v, 0)), invQuaternion(q)).xyz;
}

Transform invTransform(Transform t)
{
    t.rot = invQuaternion(t.rot);
    t.pos = -(rotateVec3(t.rot, t.pos));
    return t;
}

Transform multTransforms(Transform t1, Transform t2)
{
    Transform t3;
    t3.pos = rotateVec3(t1.rot, t2.pos);
    t3.rot = multQuaternions(t1.rot, t2.rot);
    t3.pos = t3.pos + t1.pos;
    return t3;
}

vec3 displaceVec3(Transform t, vec3 v)
{
    return rotateVec3(t.rot, v) + t.pos;
}

)"""";
