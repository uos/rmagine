#pragma once

#include <string>



/**
 * this code is preceeded by the util_preamble
 * then the define SPHERE, PINHOLE, O1DN or ONDN gets set
 * then the util_code gets included
 * 
 * util_preamble & util_code can be found in: "rmagine/shaders/VulkanIncludeShader.hpp"
 */
static const std::string rgen_code = R""""(



layout(location = 0) rayPayloadEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;


layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;


#if defined(SPHERE)
    layout(binding = 2, set = 0) buffer Sensor
    {
        DiscreteInterval phi;
        DiscreteInterval theta;
        Interval range;
    } sensor;
#elif defined(PINHOLE)
    layout(binding = 2, set = 0) buffer Sensor
    {
        int width;
        int height;
        Interval range;
        vec2 focalLength;
        vec2 center;
    } sensor;
#elif defined(O1DN)
    layout(binding = 2, set = 0) buffer Sensor
    {
        int width;
        int height;
        Interval range;
        vec3 origin;
        Memory dirsMem;
    } sensor;
#elif defined(ONDN)
    layout(binding = 2, set = 0) buffer Sensor
    {
        int width;
        int height;
        Interval range;
        Memory origsMem;
        Memory dirsMem;
    } sensor;
#else
    #error One of the the sensor types has to be defined // compile time error
#endif


layout(binding = 4, set = 0) buffer TransformBuffer{ Transform tsb; } tsb;


struct OrigsDirsAndTransforms
{
    uint64_t tbmAddress;
    
    uint64_t origsAddress;
    uint64_t dirsAddress;
};

layout(binding = 5, set = 0) buffer OrigsDirsAndTransformsBuffer{ OrigsDirsAndTransforms data; } origsDirsAndTransforms;

layout(buffer_reference, std430, buffer_reference_align = 32) buffer transform_array 
{
    Transform t;
};



vec3 getRayDir()
{
    #if defined(SPHERE)
        float phi = sensor.phi.angleMin + float(gl_LaunchIDEXT.y) * sensor.phi.angleInc;
        float theta = sensor.theta.angleMin + float(gl_LaunchIDEXT.x) * sensor.theta.angleInc;
        return vec3(cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi));
    #elif defined(PINHOLE)
        float pX = (float(gl_LaunchIDEXT.x) - sensor.center.x) / sensor.focalLength.x;
        float pY = (float(gl_LaunchIDEXT.y) - sensor.center.y) / sensor.focalLength.y;
        vec3 dirOptical = normalize(vec3(pX, pY, 1.0));
        return vec3(dirOptical.z, -dirOptical.x, -dirOptical.y);
    #elif defined(O1DN)
        float_array dirs_buffer = float_array(origsDirsAndTransforms.data.dirsAddress);
        uint index = 3*(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x);
        return vec3(dirs_buffer[index].f, dirs_buffer[index+1].f, dirs_buffer[index+2].f);
    #elif defined(ONDN)
        float_array dirs_buffer = float_array(origsDirsAndTransforms.data.dirsAddress);
        uint index = 3*(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x);
        return vec3(dirs_buffer[index].f, dirs_buffer[index+1].f, dirs_buffer[index+2].f);
    #else
        #error One of the the sensor types has to be defined! // compile time error
    #endif
}

Transform getRayStartTf()
{
    transform_array tbm = transform_array(origsDirsAndTransforms.data.tbmAddress);

    Transform sensorTf = multTransforms(tbm[gl_LaunchIDEXT.z].t, tsb.tsb);
    payload.sensorTf = sensorTf;
    Transform rayStartTf;
    rayStartTf.rot = sensorTf.rot;

    #if defined(SPHERE)
        rayStartTf.pos = sensorTf.pos;
    #elif defined(PINHOLE)
        rayStartTf.pos = sensorTf.pos;
    #elif defined(O1DN)
        rayStartTf.pos = sensorTf.pos + sensor.origin;
    #elif defined(ONDN)
        float_array origs_buffer = float_array(origsDirsAndTransforms.data.origsAddress);
        uint index = 3*(gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x);
        rayStartTf.pos = sensorTf.pos + vec3(origs_buffer[index].f, origs_buffer[index+1].f, origs_buffer[index+2].f);
    #else
        #error One of the the sensor types has to be defined! // compile time error
    #endif

    return rayStartTf;
}



void main()
{
    //get ray starting position
    Transform tsm = getRayStartTf();

    //get ray direction
    vec3 rayDirS = getRayDir();
    vec3 rayDirM = rotateVec3(tsm.rot, rayDirS);

    //trace ray
    //         (AS        | flags                         | shader binding table     | ray data                                                          | 0)
    //         (----------+---------------------+---------+-------+-------+----------+----------+--------------------+--------------+--------------------+--)
    //         (topLevelAS| flags               | cullmask| offset| stride| missIndex| rayOrigin| rayMin             | ray Direction| rayMax             | 0)
    //         (----------+---------------------+---------+-------+-------+----------+----------+--------------------+--------------+--------------------+--)
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF    , 0     , 0     , 0        , tsm.pos  , sensor.range.rayMin, rayDirM      , sensor.range.rayMax, 0);
}

)"""";
