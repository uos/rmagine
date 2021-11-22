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
 * OptixMap.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef IMAGINE_OPTIX_MAP_HPP
#define IMAGINE_OPTIX_MAP_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>
#include "AssimpMap.hpp"
#include "Map.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <imagine/types/MemoryCuda.hpp>

#include <imagine/math/types.h>

namespace imagine {

// TODO: move somewhere else
struct Face {
    unsigned int v0;
    unsigned int v1;
    unsigned int v2;
};

struct OptixAccelerationStructure
{
    OptixTraversableHandle      handle;
    CUdeviceptr                 buffer;
};

/**
 * @brief Single mesh. 
 * - Cuda Buffers for vertices, faces, normals
 * - TraversableHandle for raytracing
 */
struct OptixMesh {
    // Handle of geometry acceleration structure
    
    
    Memory<Point, VRAM_CUDA>    vertices;
    Memory<Face, VRAM_CUDA>     faces;
    Memory<Vector, VRAM_CUDA>   normals;

    // GAS
    OptixAccelerationStructure  gas;
};

using OptixMeshPtr = std::shared_ptr<OptixMesh>;

/**
 * @brief One instance is a transformed version of a mesh
 * Multiple instances can connect to a single mesh to save memory
 * - Example: asteroids. Only one mesh for a lot of instances
 * 
 */
// struct OptixInstance {
//     Transform T; // Or Matrix4x4 ?
//     OptixMeshPtr mesh; // connect a mesh to a instance
// };

using OptixInstancePtr = std::shared_ptr<OptixInstance>;

class OptixMap : public Map {
public:
    OptixMap(const aiScene* ascene, int device = 0);

    ~OptixMap();

    // OptixTraversableHandle gas_handle;
    OptixDeviceContext context = nullptr;
    int m_device;

    // TODO: make own cuda context class
    CUcontext cuda_context;

    Memory<float3, VRAM_CUDA> normals;

    std::vector<OptixMesh> meshes;
    std::vector<OptixInstance> instances;

    // Top Level AS. If loaded map consists of one mesh -> GAS. Else IAS
    OptixAccelerationStructure as;

    bool ias() const
    {
        return m_instance_level;
    }

private:

    void buildGAS(const OptixMesh& mesh, OptixAccelerationStructure& gas);

    CUdeviceptr             m_vertices = 0;
    unsigned int            m_num_vertices;
    CUdeviceptr             m_faces = 0;
    unsigned int            m_num_faces;

    bool                    m_instance_level;

    void initContext(int device = 0);

    // CUdeviceptr            d_gas_output_buffer;
};

using OptixMapPtr = std::shared_ptr<OptixMap>;

static OptixMapPtr importOptixMap(const std::string& meshfile, int device = 0)
{
    Assimp::Importer importer;
    // aiProcess_GenNormals does not work!
    const aiScene* scene = importer.ReadFile(meshfile, 0);

    if(!scene)
    {
        std::cerr << importer.GetErrorString() << std::endl;
    }

    if(!scene->HasMeshes())
    {
        std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
    }

    OptixMapPtr map(new imagine::OptixMap(scene, device) );
    return map;
}

} // namespace imagine

#endif // IMAGINE_OPTIX_MAP_HPP