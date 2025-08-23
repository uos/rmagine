#pragma once

#include <string>



/**
 * this code is preceeded by the util_preamble
 * then the defines HITS, RANGES, POINTS, NORMALS, PRIMITIVE_ID, GEOMETRY_ID & INSTANCE_ID get set
 * then the util_code gets included
 * 
 * util_preamble & util_code can be found in: "rmagine/shaders/VulkanIncludeShader.hpp"
 */
static const std::string miss_code = R""""(


layout(location = 0) rayPayloadInEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;


layout(binding = 3, set = 0) buffer ResultsBuffer{ Result data; } resultsBuffer;



void main()
{
    uint rayIndex = gl_LaunchIDEXT.z * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
                  + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
                  + gl_LaunchIDEXT.x;


    //write data to buffer(s)
    #if defined(HITS)
        //0
        uint8_array hits_buffer = uint8_array(resultsBuffer.data.hits.bufferDeviceAddress);
        hits_buffer[rayIndex].i = uint8_t(0);
    #endif
    #if defined(RANGES)
        //gl_RayTmaxEXT
        float_array ranges_buffer = float_array(resultsBuffer.data.ranges.bufferDeviceAddress);
        ranges_buffer[rayIndex].f = gl_RayTmaxEXT + 1.0;
    #endif
    #if defined(POINTS)
        //vec3(0,0,0)
        float_array points_buffer = float_array(resultsBuffer.data.points.bufferDeviceAddress);
        points_buffer[3*rayIndex    ].f = intBitsToFloat(int(0x7fffffffU));
        points_buffer[3*rayIndex + 1].f = intBitsToFloat(int(0x7fffffffU));
        points_buffer[3*rayIndex + 2].f = intBitsToFloat(int(0x7fffffffU));
    #endif
    #if defined(NORMALS)
        //vec3(0,0,0)
        float_array normals_buffer = float_array(resultsBuffer.data.normals.bufferDeviceAddress);
        normals_buffer[3*rayIndex    ].f = intBitsToFloat(int(0x7fffffffU));
        normals_buffer[3*rayIndex + 1].f = intBitsToFloat(int(0x7fffffffU));
        normals_buffer[3*rayIndex + 2].f = intBitsToFloat(int(0x7fffffffU));
    #endif
    #if defined(PRIMITIVE_ID)
        //-1
        uint_array primitiveID_buffer = uint_array(resultsBuffer.data.primitiveID.bufferDeviceAddress);
        primitiveID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
    #if defined(GEOMETRY_ID)
        //-1
        uint_array geometryID_buffer = uint_array(resultsBuffer.data.geometryID.bufferDeviceAddress);
        geometryID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
    #if defined(INSTANCE_ID)
        //-1
        uint_array instanceID_buffer = uint_array(resultsBuffer.data.instanceID.bufferDeviceAddress);
        instanceID_buffer[rayIndex].i = uint(0xffffffffU);
    #endif
}

)"""";
