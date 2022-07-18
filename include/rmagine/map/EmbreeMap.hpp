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

#include "Map.hpp"
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


namespace rmagine {

class EmbreeDevice;
class EmbreeScene;
class EmbreeMesh;
class EmbreeInstance;

using EmbreeDevicePtr = std::shared_ptr<EmbreeDevice>;
using EmbreeScenePtr = std::shared_ptr<EmbreeScene>;
using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>; 
using EmbreeInstancePtr = std::shared_ptr<EmbreeInstance>;


using EmbreeInstanceSet = std::unordered_set<EmbreeInstancePtr>;

class EmbreeDevice
{
public:
    EmbreeDevice();

    ~EmbreeDevice();

    RTCDevice handle();

private:
    RTCDevice m_device;
};

class EmbreeScene
{
public:
    EmbreeScene(EmbreeDevicePtr device)
    {
        m_scene = rtcNewScene(device->handle());
    }

    ~EmbreeScene()
    {
        rtcReleaseScene(m_scene);
    }

    RTCScene handle()
    {
        return m_scene;
    }

    void commit()
    {
        rtcCommitScene(m_scene);
    }

private:
    RTCScene m_scene;
};

class EmbreeMesh
{
public:
    // TODO: constructor destructor

    EmbreeMesh( EmbreeDevicePtr device);

    EmbreeMesh( EmbreeDevicePtr device, 
                unsigned int Nvertices, 
                unsigned int Nfaces);

    EmbreeMesh( EmbreeDevicePtr device,
                const aiMesh* amesh);

    // embree constructed buffers
    unsigned int Nvertices;
    Vertex* vertices;

    unsigned int Nfaces;
    Face* faces;
    
    // more custom attributes
    Memory<Vector, RAM> normals;

    // embree fields
    RTCGeometry handle;
    unsigned int geomID;

    void transform(const Matrix4x4& T);
    
    void setScene(EmbreeScenePtr scene);
    void setNewScene();
    EmbreeScenePtr scene();

    void addInstance(EmbreeInstancePtr instance);
    bool hasInstance(EmbreeInstancePtr instance) const;
    EmbreeInstanceSet instances();

    void commit();
private:
    // connections
    EmbreeInstanceSet m_instances;
    EmbreeScenePtr m_scene;
    EmbreeDevicePtr m_device;
};

class EmbreeInstance
{
public:
    Matrix4x4 T;

    // embree fields
    RTCGeometry handle;
    unsigned int instID;

    void setScene(EmbreeScenePtr scene);
    EmbreeScenePtr scene();

    void setMesh(EmbreeMeshPtr mesh);
    EmbreeMeshPtr mesh();

    // Make this more comfortable to use
    // - functions as: setMesh(), or addMesh() ?
    // - translate rotate scale? 

    /**
     * @brief Call update after changing the transformation. TODO TEST
     * 
     */
    void commit();

private:

    EmbreeMeshPtr m_mesh;
    EmbreeScenePtr m_scene;
};

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

class EmbreeMap : public Map {
public:
    EmbreeMap();
    EmbreeMap(const aiScene* ascene);
    ~EmbreeMap();

    unsigned int addMesh(EmbreeMeshPtr mesh);

    Point closestPoint(const Point& qp);

    EmbreeDevicePtr device;
    EmbreeScenePtr scene;
    // TODO:

    std::vector<EmbreeMeshPtr> meshes;
    std::vector<EmbreeInstancePtr> instances;

    RTCPointQueryContext pq_context;

protected:
    std::vector<EmbreeMeshPtr> loadMeshes(const aiScene* ascene);

    std::vector<EmbreeInstancePtr> loadInstances(
        const aiNode* root_node,
        std::vector<EmbreeMeshPtr>& meshes);
};

using EmbreeMapPtr = std::shared_ptr<EmbreeMap>;

static EmbreeMapPtr importEmbreeMap(const std::string& meshfile)
{
    Assimp::Importer importer;
    // aiProcess_GenNormals does not work!
    const aiScene* scene = importer.ReadFile( meshfile, 0);

    if(!scene)
    {
        std::cerr << importer.GetErrorString() << std::endl;
    }

    if(!scene->HasMeshes())
    {
        std::cerr << "[RMagine - Error] importEmbreeMap() - file '" << meshfile << "' contains no meshes" << std::endl;
    }

    EmbreeMapPtr map(new EmbreeMap(scene) );
    return map;
}

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MAP_HPP