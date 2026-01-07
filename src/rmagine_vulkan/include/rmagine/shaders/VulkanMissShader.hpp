#pragma once

#include <string>



/**
 * miss shader
 * this code is preceeded by the util_code, which can be found in: "rmagine/shaders/VulkanIncludeShader.hpp"
 * then the defines HITS, RANGES, POINTS, NORMALS, PRIMITIVE_ID, GEOMETRY_ID & INSTANCE_ID get set
 */
static const std::string miss_code = R""""(
#if !defined(HITS) && !defined(RANGES) && !defined(POINTS) && !defined(NORMALS) && !defined(PRIMITIVE_ID) && !defined(GEOMETRY_ID) && !defined(INSTANCE_ID)
    #error At least one of the result types has to be defined // compile time error
#endif





layout(location = 0) rayPayloadInEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;


layout(binding = 3, set = 0) uniform ResultsBuffer{ Result data; } resultsBuffer;



void main()
{
    uint rayIndex = gl_LaunchIDEXT.z * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
                  + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
                  + gl_LaunchIDEXT.x;


    //write data to buffer(s)
    #if defined(HITS)
        //0
        uint8_array hits_buffer = uint8_array(resultsBuffer.data.hitsAddress);
        hits_buffer[rayIndex].i = uint8_t(0);
    #endif
    #if defined(RANGES)
        //gl_RayTmaxEXT
        float_array ranges_buffer = float_array(resultsBuffer.data.rangesAddress);
        ranges_buffer[rayIndex].f = gl_RayTmaxEXT + 1.0;
    #endif
    #if defined(POINTS)
        //vec3(0,0,0)
        float_array points_buffer = float_array(resultsBuffer.data.pointsAddress);
        points_buffer[3*rayIndex    ].f = intBitsToFloat(int(0x7fffffffU));
        points_buffer[3*rayIndex + 1].f = intBitsToFloat(int(0x7fffffffU));
        points_buffer[3*rayIndex + 2].f = intBitsToFloat(int(0x7fffffffU));
    #endif
    #if defined(NORMALS)
        //vec3(0,0,0)
        float_array normals_buffer = float_array(resultsBuffer.data.normalsAddress);
        normals_buffer[3*rayIndex    ].f = intBitsToFloat(int(0x7fffffffU));
        normals_buffer[3*rayIndex + 1].f = intBitsToFloat(int(0x7fffffffU));
        normals_buffer[3*rayIndex + 2].f = intBitsToFloat(int(0x7fffffffU));
    #endif
    #if defined(PRIMITIVE_ID)
        //-1
        uint_array primitiveID_buffer = uint_array(resultsBuffer.data.primitiveIdAddress);
        primitiveID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
    #if defined(GEOMETRY_ID)
        //-1
        uint_array geometryID_buffer = uint_array(resultsBuffer.data.geometryIdAddress);
        geometryID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
    #if defined(INSTANCE_ID)
        //-1
        uint_array instanceID_buffer = uint_array(resultsBuffer.data.instanceIdAddress);
        instanceID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
}

)"""";
