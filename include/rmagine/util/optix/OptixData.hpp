/**
 * Copyright (c) 2021, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * OptixData.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_OPTIX_DATA_HPP
#define RMAGINE_OPTIX_DATA_HPP

#include <cuda_runtime.h>
#include <rmagine/math/types.h>
#include <rmagine/types/sensor_models.h>

#include <rmagine/map/optix/optix_definitions.h>


namespace rmagine 
{

struct RayGenDataEmpty {

};

// For fixed spherical model
// struct RayGenDataSpherical {
//     SphericalModel model;
// };

struct MissDataEmpty {

};

struct HitGroupDataEmpty {

};

struct HitGroupDataNormals {
    // instance -> normals
    Vector** normals;
};

struct MeshAttributes 
{
    Vector* vertex_normals;
    Vector* face_normals;
};


struct InstanceAttributes
{
    // one instance can have multiple geometries
    int* geom_ids;
    // global mesh ids
    int* mesh_ids;
};

struct HitGroupDataScene {
    // instance attributes
    unsigned int n_instances = 0;
    InstanceAttributes* instances_attributes = nullptr;

    // link from instance id to mesh id
    // int* inst_to_mesh = nullptr;

    // mesh attributes
    unsigned int n_meshes = 0;
    MeshAttributes* mesh_attributes = nullptr;
};

// TODO MOVE TO MAP

// FORWARD DECLARE
// union OptixGeomSBT;
// struct OptixSceneSBT;
// struct OptixMeshSBT;
// struct OptixInstanceSBT;


// struct OptixMeshSBT
// {
//     Vector* vertex_normals = nullptr;
//     Vector* face_normals = nullptr;
//     unsigned int id = 0;
// };

// struct OptixInstanceSBT
// {
//     OptixSceneSBT* scene = nullptr;
// };

// union OptixGeomSBT
// {
//     OptixMeshSBT mesh_data;
//     OptixInstanceSBT inst_data;
// };

// struct OptixSceneSBT
// {
//     OptixSceneType type;
//     unsigned int n_geometries = 0;
//     OptixGeomSBT* geometries = nullptr;
// };

} // namespace rmagine

#endif // RMAGINE_OPTIX_DATA_HPP