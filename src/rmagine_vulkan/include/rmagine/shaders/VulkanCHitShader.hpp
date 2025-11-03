#pragma once

#include <string>



/**
 * closest hit shader
 * this code is preceeded by the util_code, which can be found in: "rmagine/shaders/VulkanIncludeShader.hpp"
 * then the defines HITS, RANGES, POINTS, NORMALS, PRIMITIVE_ID, GEOMETRY_ID & INSTANCE_ID get set
 */
static const std::string chit_code = R""""(
#if !defined(HITS) && !defined(RANGES) && !defined(POINTS) && !defined(NORMALS) && !defined(PRIMITIVE_ID) && !defined(GEOMETRY_ID) && !defined(INSTANCE_ID)
    #error At least one of the result types has to be defined // compile time error
#endif


hitAttributeEXT vec2 hitCoordinate;


layout(location = 0) rayPayloadInEXT Payload
{
    //needed for transforming to sensor coordinates
    Transform sensorTf;
} payload;


layout(binding = 3, set = 0) uniform ResultsBuffer{ Result data; } resultsBuffer;


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
    #if defined(POINTS) || defined(NORMALS)
        Transform sensorTf_inv = invTransform(payload.sensorTf);

        vec3 ray_dir_s = rotateVec3(sensorTf_inv.rot, gl_WorldRayDirectionEXT);
    #endif


    #if defined(POINTS)
        vec3 ray_pos_s = displaceVec3(sensorTf_inv, gl_WorldRayOriginEXT);

        vec3 position = ray_pos_s + (gl_HitTEXT * ray_dir_s);
    #endif


    #if defined(NORMALS)
        //normal in object space
        vec3 normal_object;

        meshDesc_array meshDescs = meshDesc_array(mapDataBuffer.data[gl_InstanceID]);

        // summary:
        // use (interpolated) vertex normals if possible
        // else use face normals if possible
        // else calculate face normals yourself using verticies and indicies

        if(meshDescs[gl_GeometryIndexEXT].meshDesc.vertexNormalAddress != 0)
        {
            //interpolate vertex normals to get hit normal

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
            
            normal_object = (barycentric.x * vertexNormalA) + (barycentric.y * vertexNormalB) + (barycentric.z * vertexNormalC);
        }
        else if(meshDescs[gl_GeometryIndexEXT].meshDesc.faceNormalAddress != 0)
        {
            //grab face normal from array

            float_array faceNormals = float_array(meshDescs[gl_GeometryIndexEXT].meshDesc.faceNormalAddress);

            normal_object = vec3(faceNormals[3 * gl_PrimitiveID + 0].f,
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

            normal_object = normalize(cross(vertexB - vertexA, vertexC - vertexA));
        }

        //transform normal from object to world space
        vec3 normal_world = normalize(mat3(gl_ObjectToWorldEXT) * normal_object);

        //transform normal from world to sensor space
        vec3 normal = rotateVec3(sensorTf_inv.rot, normal_world);

        //flip?
        if(dot(ray_dir_s, normal) > 0.0)
        {
            normal *= -1.0;
        }
    #endif


    uint rayIndex = gl_LaunchIDEXT.z * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
                  + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
                  + gl_LaunchIDEXT.x;


    //write data to buffer(s)
    #if defined(HITS)
        //1
        uint8_array hits_buffer = uint8_array(resultsBuffer.data.hitsAddress);
        hits_buffer[rayIndex].i = uint8_t(1);
    #endif
    #if defined(RANGES)
        //gl_HitTEXT
        float_array ranges_buffer = float_array(resultsBuffer.data.rangesAddress);
        ranges_buffer[rayIndex].f = gl_HitTEXT;
    #endif
    #if defined(POINTS)
        //position
        float_array points_buffer = float_array(resultsBuffer.data.pointsAddress);
        points_buffer[3*rayIndex    ].f = position.x;
        points_buffer[3*rayIndex + 1].f = position.y;
        points_buffer[3*rayIndex + 2].f = position.z;
    #endif
    #if defined(NORMALS)
        //normal
        float_array normals_buffer = float_array(resultsBuffer.data.normalsAddress);
        normals_buffer[3*rayIndex    ].f = normal.x;
        normals_buffer[3*rayIndex + 1].f = normal.y;
        normals_buffer[3*rayIndex + 2].f = normal.z;
    #endif
    #if defined(PRIMITIVE_ID)
        //gl_PrimitiveID
        uint_array primitiveID_buffer = uint_array(resultsBuffer.data.primitiveIdAddress);
        primitiveID_buffer[rayIndex].i = uint(gl_PrimitiveID);
    #endif
    #if defined(GEOMETRY_ID)
        //gl_GeometryIndexEXT
        uint_array geometryID_buffer = uint_array(resultsBuffer.data.geometryIdAddress);
        geometryID_buffer[rayIndex].i = uint(gl_GeometryIndexEXT);
    #endif
    #if defined(INSTANCE_ID)
        //gl_InstanceID
        uint_array instanceID_buffer = uint_array(resultsBuffer.data.instanceIdAddress);
        instanceID_buffer[rayIndex].i = uint(gl_InstanceCustomIndexEXT); // uint(gl_InstanceID);
    #endif
}

)"""";
