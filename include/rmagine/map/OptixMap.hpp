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

#include "optix/OptixAccelerationStructure.hpp"

#include "optix/OptixMesh.hpp"

#include "optix/OptixScene.hpp"

namespace rmagine {

class OptixMap 
: public OptixEntity {
public:
    OptixMap(OptixScenePtr scene);

    OptixMap(OptixContextPtr optix_ctx = optix_default_context());

    virtual ~OptixMap();

    void setScene(OptixScenePtr scene);
    OptixScenePtr scene() const;

protected:
    OptixScenePtr m_scene;
};

using OptixMapPtr = std::shared_ptr<OptixMap>;

/**
 * @brief Import a mesh file as OptixMap to an existing OptixContext 
 * 
 */
static OptixMapPtr importOptixMap(
    const std::string& meshfile, 
    OptixContextPtr optix_ctx = optix_default_context())
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
        std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
    }

    OptixScenePtr scene = make_optix_scene(ascene, optix_ctx);
    return std::make_shared<OptixMap>(scene);
}

} // namespace rmagine

#endif // RMAGINE_OPTIX_MAP_HPP