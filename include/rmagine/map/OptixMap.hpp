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

#ifndef RMAGINE_OPTIX_MAP_HPP
#define RMAGINE_OPTIX_MAP_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>
#include "AssimpMap.hpp"
#include "Map.hpp"


#include <rmagine/types/MemoryCuda.hpp>

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <unordered_map>

#include <rmagine/util/cuda/CudaContext.hpp>
#include <rmagine/util/optix/OptixContext.hpp>
#include "AssimpIO.hpp"

namespace rmagine {
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
    unsigned int id;

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
    /**
     * @brief Construct a new Optix Map object. 
     *   A new Optix and Cuda context on device is created an can be later accessed by calling context()
     * 
     * @param ascene 
     * @param device 
     */
    OptixMap(
        const aiScene* ascene, 
        int device = 0);

    /**
     * @brief Construct a new Optix Map object
     * 
     * @param ascene 
     * @param optix_ctx 
     */
    OptixMap(
        const aiScene* ascene, 
        OptixContextPtr optix_ctx);

    ~OptixMap();

    // OptixDeviceContext context = nullptr;
    // int m_device;

    // TODO: make own cuda context class
    // CUcontext cuda_context;

    // CudaContextPtr m_cuda_context;

    Memory<OptixInstance, RAM> instances;
    std::vector<OptixMesh> meshes;

    /** top-level structure: can be one of:
    * - Geometry Acceleration Structure (GAS). If loaded map contains only one mesh
    * - Instance Acceleration Structure (IAS). If loaded map contrains more than one mesh. 
    * 
    * Current Implementation: Each Instance has exactly one mesh (TODO)
    *
    */ 
    OptixAccelerationStructure as;

    /**
     * @brief If the map has a instance level acceleration structure
     * 
     * @return true Yes
     * @return false No
     */
    bool ias() const
    {
        return m_instance_level;
    }

    OptixContextPtr context() const
    {
        return m_optix_context;
    }

private:

    void buildStructures(const aiScene* ascene);

    // BUILD GAS
    void fillMeshes(const aiScene* ascene);

    void buildGAS(
        const OptixMesh& mesh, 
        OptixAccelerationStructure& gas);

    // BUILD IAS
    // fillInstances, buildIAS requires fillMeshes and buildGAS to be called first
    void fillInstances(const aiScene* ascene);

    void buildIAS(
        const Memory<OptixInstance, VRAM_CUDA>& instances,
        OptixAccelerationStructure& ias
    );

    void buildIAS(
        const Memory<OptixInstance, RAM>& instances,
        OptixAccelerationStructure& ias);

    bool                    m_instance_level;

    OptixContextPtr         m_optix_context;
};

using OptixMapPtr = std::shared_ptr<OptixMap>;

/**
 * @brief 
 * 
 * @param meshfile 
 * @param device 
 * @return OptixMapPtr 
 */
static OptixMapPtr importOptixMap(
    const std::string& meshfile, 
    int device = 0)
{
    AssimpIO io;
    // aiProcess_GenNormals does not work!
    const aiScene* scene = io.ReadFile(meshfile, 0);

    if(!scene)
    {
        std::cerr << io.Import::GetErrorString() << std::endl;
    }

    if(!scene->HasMeshes())
    {
        std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
    }

    OptixMapPtr map(new OptixMap(scene, device) );
    return map;
}

/**
 * @brief Import a mesh file as OptixMap to an existing OptixContext 
 * 
 */
static OptixMapPtr importOptixMap(
    const std::string& meshfile, OptixContextPtr optix_ctx)
{
    AssimpIO io;
    // aiProcess_GenNormals does not work!
    const aiScene* scene = io.ReadFile(meshfile, 0);

    if(!scene)
    {
        std::cerr << io.Import::GetErrorString() << std::endl;
    }

    if(!scene->HasMeshes())
    {
        std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
    }

    OptixMapPtr map(new OptixMap(scene, optix_ctx) );
    return map;
}

} // namespace rmagine

#endif // RMAGINE_OPTIX_MAP_HPP