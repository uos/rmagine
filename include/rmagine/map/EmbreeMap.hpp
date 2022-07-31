/*
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

/**
 * @file
 * 
 * @brief Contains @link rmagine::EmbreeMap EmbreeMap @endlink
 *
 * @date 03.01.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2021, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_MAP_HPP
#define RMAGINE_MAP_EMBREE_MAP_HPP

#include <embree3/rtcore.h>

#include "AssimpMap.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>

#include <rmagine/math/types.h>
#include <rmagine/math/math.h>
#include <rmagine/types/mesh_types.h>

#include <rmagine/types/Memory.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/assimp_conversions.h>
#include "AssimpIO.hpp"

#include "embree/EmbreeDevice.hpp"
#include "embree/EmbreeScene.hpp"
#include "embree/EmbreeMesh.hpp"
#include "embree/EmbreeInstance.hpp"


namespace rmagine 
{

struct ClosestPointResult
{
    ClosestPointResult() 
        : d(std::numeric_limits<float>::max())
        , primID(RTC_INVALID_GEOMETRY_ID)
        , geomID(RTC_INVALID_GEOMETRY_ID)
    {}

    float d;
    Point p;
    unsigned int primID;
    unsigned int geomID;
};

struct PointQueryUserData 
{
    std::vector<EmbreeMesh>* parts;
    ClosestPointResult* result;
};

class EmbreeMap {
public:
    EmbreeMap(EmbreeDevicePtr device = embree_default_device());
    EmbreeMap(const aiScene* ascene, EmbreeDevicePtr device = embree_default_device());
    
    ~EmbreeMap();

    void set(const aiScene* ascene);

    unsigned int addMesh(EmbreeMeshPtr mesh);

    Point closestPoint(const Point& qp);

    EmbreeDevicePtr device;
    EmbreeScenePtr scene;
    // TODO:

    std::unordered_set<EmbreeMeshPtr> meshes;

    // std::vector<EmbreeMeshPtr> meshes;
    // std::vector<EmbreeInstancePtr> instances;

    RTCPointQueryContext pq_context;

protected:
    std::vector<EmbreeMeshPtr> loadMeshes(const aiScene* ascene);

    std::vector<EmbreeInstancePtr> loadInstances(
        const aiNode* root_node,
        std::vector<EmbreeMeshPtr>& meshes);
};

using EmbreeMapPtr = std::shared_ptr<EmbreeMap>;

static EmbreeMapPtr importEmbreeMap(
    const std::string& meshfile,
    EmbreeDevicePtr device = embree_default_device())
{
    std::cout << "importEmbreeMap - WARNING deprecated" << std::endl;
    AssimpIO io;

    // aiProcess_GenNormals does not work!
    const aiScene* scene = io.ReadFile( meshfile, 0);

    if(!scene)
    {
        std::cerr << io.Import::GetErrorString() << std::endl;
    }

    if(!scene->HasMeshes())
    {
        std::cerr << "[RMagine - Error] importEmbreeMap() - file '" << meshfile << "' contains no meshes" << std::endl;
    }

    return std::make_shared<EmbreeMap>(scene, device);
}

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MAP_HPP