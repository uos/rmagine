#pragma once

#include <string>



static const std::string chit_preamble = R""""(#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_ARB_shading_language_include : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
)"""";



static const std::string chit_code = R""""(
//the defines HITS, RANGES, POINTS, NORMALS, PRIMITIVE_ID, GEOMETRY_ID & INSTANCE_ID get set during compilation



hitAttributeEXT vec2 hitCoordinate;

layout(location = 0) rayPayloadInEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;

layout(binding = 4, set = 0) buffer ResultsBuffer{ Result data; } resultsBuffer;

layout(binding = 2, set = 0) buffer IndexBuffer { uint data[]; } indexBuffer;
layout(binding = 3, set = 0) buffer VertexBuffer { float data[]; } vertexBuffer;



void main()
{
    #if defined(POINTS) || defined(NORMALS)
        ivec3 indices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0],
                              indexBuffer.data[3 * gl_PrimitiveID + 1],
                              indexBuffer.data[3 * gl_PrimitiveID + 2]);

        vec3 vertexA = vec3(vertexBuffer.data[3 * indices.x + 0],
                            vertexBuffer.data[3 * indices.x + 1],
                            vertexBuffer.data[3 * indices.x + 2]);
        vec3 vertexB = vec3(vertexBuffer.data[3 * indices.y + 0],
                            vertexBuffer.data[3 * indices.y + 1],
                            vertexBuffer.data[3 * indices.y + 2]);
        vec3 vertexC = vec3(vertexBuffer.data[3 * indices.z + 0],
                            vertexBuffer.data[3 * indices.z + 1],
                            vertexBuffer.data[3 * indices.z + 2]);
    #endif

    #if defined(POINTS)
        vec3 barycentric = vec3(1.0 - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);

        vec3 position = vertexA * barycentric.x + vertexB * barycentric.y + vertexC * barycentric.z;

        vec3 positionTransfromed = rotateVec3(payload.sensorTf.rot, (position - payload.sensorTf.pos));//CHECK: test if math is correct...
    #endif

    #if defined(NORMALS)
        vec3 geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));
    #endif


    uint rayIndex = gl_LaunchIDEXT.z * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
                  + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
                  + gl_LaunchIDEXT.x;


    //write data to buffer(s)
    #if defined(HITS)
        //1
        uint8_array hits_buffer = uint8_array(resultsBuffer.data.hits.bufferDeviceAddress);
        hits_buffer[rayIndex].i = uint8_t(1);
    #endif
    #if defined(RANGES)
        //gl_HitTEXT
        float_array ranges_buffer = float_array(resultsBuffer.data.ranges.bufferDeviceAddress);
        ranges_buffer[rayIndex].f = gl_HitTEXT;
    #endif
    #if defined(POINTS)
        //positionTransfromed
        float_array points_buffer = float_array(resultsBuffer.data.points.bufferDeviceAddress);
        points_buffer[3*rayIndex    ].f = positionTransfromed.x;
        points_buffer[3*rayIndex + 1].f = positionTransfromed.y;
        points_buffer[3*rayIndex + 2].f = positionTransfromed.z;
    #endif
    #if defined(NORMALS)
        //geometricNormal
        float_array normals_buffer = float_array(resultsBuffer.data.normals.bufferDeviceAddress);
        normals_buffer[3*rayIndex    ].f = geometricNormal.x;
        normals_buffer[3*rayIndex + 1].f = geometricNormal.y;
        normals_buffer[3*rayIndex + 2].f = geometricNormal.z;
    #endif
    #if defined(PRIMITIVE_ID)
        //gl_PrimitiveID
        uint_array primitiveID_buffer = uint_array(resultsBuffer.data.primitiveID.bufferDeviceAddress);
        primitiveID_buffer[rayIndex].i = uint(gl_PrimitiveID);
    #endif
    #if defined(GEOMETRY_ID)
        //gl_GeometryIndexEXT
        uint_array geometryID_buffer = uint_array(resultsBuffer.data.geometryID.bufferDeviceAddress);
        geometryID_buffer[rayIndex].i = uint(gl_GeometryIndexEXT);
    #endif
    #if defined(INSTANCE_ID)
        //gl_InstanceID
        uint_array instanceID_buffer = uint_array(resultsBuffer.data.instanceID.bufferDeviceAddress);
        instanceID_buffer[rayIndex].i = uint(gl_InstanceID);
    #endif
}

)"""";
