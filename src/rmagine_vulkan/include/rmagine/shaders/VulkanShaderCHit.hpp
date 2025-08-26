#pragma once

#include <string>



/**
 * this code is preceeded by the util_preamble
 * then the defines HITS, RANGES, POINTS, NORMALS, PRIMITIVE_ID, GEOMETRY_ID & INSTANCE_ID get set
 * then the util_code gets included
 * 
 * util_preamble & util_code can be found in: "rmagine/shaders/VulkanIncludeShader.hpp"
 */
static const std::string chit_code = R""""(
hitAttributeEXT vec2 hitCoordinate;

layout(location = 0) rayPayloadInEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;


layout(binding = 3, set = 0) buffer ResultsBuffer{ Result data; } resultsBuffer;


layout(binding = 1, set = 0) buffer MapDataBuffer { uint64_t data[]; } mapDataBuffer;

struct MeshDescription
{
    uint64_t vertexAddress;
    uint64_t faceAddress;
    uint64_t faceNormalAddress;
    uint64_t vertexNormalAddress;
};

layout(buffer_reference, std430, buffer_reference_align = 32) buffer meshDesc_array
{
    MeshDescription meshDesc;
};



void main()
{
    #if defined(POINTS)
        vec3 ray_pos_s = displaceVec3(payload.sensorTf, gl_WorldRayOriginEXT);

        vec3 ray_dir_s = rotateVec3(payload.sensorTf.rot, gl_WorldRayDirectionEXT);

        vec3 positionTransfromed = ray_pos_s + (gl_HitTEXT * ray_dir_s);
    #endif


    #if defined(NORMALS)
        vec3 geometricNormal;

        meshDesc_array meshDescs = meshDesc_array(mapDataBuffer.data[gl_InstanceID]);

        if(meshDescs[gl_GeometryIndexEXT].meshDesc.vertexNormalAddress != 0)
        {
            //interpolate vertex normals to get hit normal - TODO: not tested

            uint_array  faces     =  uint_array(meshDescs[gl_GeometryIndexEXT].meshDesc.faceAddress);
            float_array vertexNormals = float_array(meshDescs[gl_GeometryIndexEXT].meshDesc.vertexNormalAddress);

            vec3 barycentric = vec3(1.0 - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);

            uvec3 indices = uvec3(faces[3 * gl_PrimitiveID + 0].i,
                                  faces[3 * gl_PrimitiveID + 1].i,
                                  faces[3 * gl_PrimitiveID + 2].i);

            vec3 vertexNormalA = vec3(vertexNormals[3 * indices.x + 0].f,
                                      vertexNormals[3 * indices.x + 1].f,
                                      vertexNormals[3 * indices.x + 2].f);
            vec3 vertexNormalB = vec3(vertexNormals[3 * indices.y + 0].f,
                                      vertexNormals[3 * indices.y + 1].f,
                                      vertexNormals[3 * indices.y + 2].f);
            vec3 vertexNormalC = vec3(vertexNormals[3 * indices.z + 0].f,
                                      vertexNormals[3 * indices.z + 1].f,
                                      vertexNormals[3 * indices.z + 2].f);
            
            geometricNormal = (barycentric.x * vertexNormalA) + (barycentric.y * vertexNormalB) + (barycentric.z * vertexNormalC);
        }
        else if(meshDescs[gl_GeometryIndexEXT].meshDesc.faceNormalAddress != 0)
        {
            //grab face normal from array

            float_array faceNormals = float_array(meshDescs[gl_GeometryIndexEXT].meshDesc.faceNormalAddress);

            geometricNormal = vec3(faceNormals[3 * gl_PrimitiveID + 0].f,
                                   faceNormals[3 * gl_PrimitiveID + 1].f,
                                   faceNormals[3 * gl_PrimitiveID + 2].f);
        }
        else
        {
            //calculate face normal from vertex and index data

            uint_array  faces     =  uint_array(meshDescs[gl_GeometryIndexEXT].meshDesc.faceAddress);
            float_array verticies = float_array(meshDescs[gl_GeometryIndexEXT].meshDesc.vertexAddress);

            uvec3 indices = uvec3(faces[3 * gl_PrimitiveID + 0].i,
                                  faces[3 * gl_PrimitiveID + 1].i,
                                  faces[3 * gl_PrimitiveID + 2].i);

            vec3 vertexA = vec3(verticies[3 * indices.x + 0].f,
                                verticies[3 * indices.x + 1].f,
                                verticies[3 * indices.x + 2].f);
            vec3 vertexB = vec3(verticies[3 * indices.y + 0].f,
                                verticies[3 * indices.y + 1].f,
                                verticies[3 * indices.y + 2].f);
            vec3 vertexC = vec3(verticies[3 * indices.z + 0].f,
                                verticies[3 * indices.z + 1].f,
                                verticies[3 * indices.z + 2].f);

            geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));
        }

        //TODO:transform normal to sensor space
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
        instanceID_buffer[rayIndex].i = uint(gl_InstanceCustomIndexEXT); // uint(gl_InstanceID); // TODO: which one do I use?
    #endif
}

)"""";
