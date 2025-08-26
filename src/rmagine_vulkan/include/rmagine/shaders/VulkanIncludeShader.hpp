#pragma once

#include <string>

//TDOD: put strings in external shaderfiles and include them with @INCLUDE_STRING@. example in cmake/FileToString.h.in

static const std::string util_preamble = R""""(#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_ARB_shading_language_include : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
)"""";



static const std::string util_code = R""""(
// the memory struct sent to the fpu contains a pointer, which is depending on CPU 32 or 64 Bit long
#ifndef _32BIT
    #define _64BIT
#endif



#if defined(O1DN) || defined(ONDN) || defined(RANGES) || defined(POINTS) || defined(NORMALS) || defined(POINTS) || defined(NORMALS)
    layout(buffer_reference, std430, buffer_reference_align = 4) buffer float_array //cant be a vec3 array as that would introduce 4 bytes of padding 
    {
        float f;
    };
#endif
#if defined(PRIMITIVE_ID) || defined(GEOMETRY_ID) || defined(INSTANCE_ID) || defined(POINTS) || defined(NORMALS)
    layout(buffer_reference, std430, buffer_reference_align = 4) buffer uint_array 
    {
        uint i;
    };
#endif
#if defined(HITS)
    layout(buffer_reference, std430, buffer_reference_align = 1) buffer uint8_array  
    {
        uint8_t i;
    };
#endif



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

struct Memory
{
    uint64_t bufferDeviceAddress;
    //the memory object contains a shared pointer which consists of 2 pointers (irrelevant1 & irrelevant2)
    #if defined(_64BIT)
        uint64_t irrelevant1;
        uint64_t irrelevant2;
    #elif defined(_32BIT)
        int irrelevant1;
        int irrelevant2;
    #endif
};

struct Result
{
    Memory hits;
    Memory ranges;
    Memory points;
    Memory normals;
    Memory primitiveID;
    Memory instanceID;
    Memory geometryID;
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
    //both versions should work, but the second version seems to be slightly faster.
    // return multQuaternions(multQuaternions(q, vec4(v, 0)), invQuaternion(q)).xyz;
    return v + 2.0*cross(cross(v, q.xyz ) + q.w*v, q.xyz);
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
