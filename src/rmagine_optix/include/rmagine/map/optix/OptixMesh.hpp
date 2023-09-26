/*
 * Copyright (c) 2022, University Osnabrück
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
 * @brief OptixMesh
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_MESH_HPP
#define RMAGINE_MAP_OPTIX_MESH_HPP

#include <cuda_runtime.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <rmagine/util/cuda/CudaContext.hpp>
#include <rmagine/util/optix/OptixContext.hpp>

#include <memory>

#include <assimp/mesh.h>

#include "OptixGeometry.hpp"

#include "optix_definitions.h"

#include "optix_sbt.h"


namespace rmagine
{

/**
//  * @brief Single mesh. 
//  * - Cuda Buffers for vertices, faces, vertex_normals and face_normals
//  * - TraversableHandle for raytracing
//  */
class OptixMesh 
: public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixMesh(OptixContextPtr context = optix_default_context());

    virtual ~OptixMesh();

    virtual void apply();
    virtual void commit();

    virtual unsigned int depth() const;

    virtual OptixGeometryType type() const 
    {
        return OptixGeometryType::MESH;
    }

    void computeFaceNormals();

    const CUdeviceptr* getVertexBuffer() const;
    CUdeviceptr getFaceBuffer();

    // TODO manage read and write access over functions
    // before transform: write here
    Memory<Point, VRAM_CUDA>    vertices;
    Memory<Face, VRAM_CUDA>     faces;
    Memory<Vector, VRAM_CUDA>   face_normals;
    Memory<Vector, VRAM_CUDA>   vertex_normals;

    float pre_transform_h[12];
    CUdeviceptr pre_transform = 0;

    OptixMeshSBT sbt_data;
private:
    CUdeviceptr m_vertices_ref;
};

using OptixMeshPtr = std::shared_ptr<OptixMesh>;



// 
OptixMeshPtr make_optix_mesh(
    const aiMesh* amesh,
    OptixContextPtr context = optix_default_context());

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_MESH_HPP