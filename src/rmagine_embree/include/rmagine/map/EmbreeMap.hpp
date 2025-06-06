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


#include <embree4/rtcore.h>

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <limits>

#include <rmagine/math/types.h>
#include <rmagine/math/math.h>
#include <rmagine/types/mesh_types.h>

#include <rmagine/map/Map.hpp>

#include <rmagine/types/Memory.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/math/assimp_conversions.h>
#include <rmagine/map/AssimpIO.hpp>

#include "embree/EmbreeDevice.hpp"
#include "embree/EmbreeScene.hpp"
#include "embree/EmbreeMesh.hpp"
#include "embree/EmbreeInstance.hpp"


namespace rmagine 
{

class EmbreeMap : public Map {
public:
    EmbreeMap(EmbreeDevicePtr device = embree_default_device());
    EmbreeMap(EmbreeScenePtr scene);

    ~EmbreeMap();

    // utility funtions
    EmbreeClosestPointResult closestPoint(
      const Point& qp, 
      const float& max_distance = std::numeric_limits<float>::max());

    EmbreeDevicePtr device;
    EmbreeScenePtr scene;

    // container for storing meshes for faster access
    // - meshes are also shared referenced somewhere in scene
    // - filling is not mandatory
    // TODO: not only meshes here. Every geometry
    std::unordered_set<EmbreeMeshPtr> meshes;
};

using EmbreeMapPtr = std::shared_ptr<EmbreeMap>;

static EmbreeMapPtr import_embree_map(
    const std::string& meshfile,
    EmbreeDevicePtr device = embree_default_device())
{
    AssimpIO io;

    // aiProcess_GenNormals does not work!
    const aiScene* ascene = io.ReadFile(meshfile, 0);

    if(!ascene)
    {
        std::cerr << io.Importer::GetErrorString() << std::endl;
    }

    if(!ascene->HasMeshes())
    {
        std::cerr << "[RMagine - Error] importEmbreeMap() - file '" << meshfile << "' contains no meshes" << std::endl;
    }

    EmbreeScenePtr scene = make_embree_scene(ascene, device);
    scene->freeze();
    scene->commit();
    return std::make_shared<EmbreeMap>(scene);
}

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MAP_HPP