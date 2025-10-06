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
 * @brief EmbreeInstance
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_INSTANCE_HPP
#define RMAGINE_MAP_EMBREE_INSTANCE_HPP

#include "embree_definitions.h"

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>
#include <unordered_set>
#include <memory>

#include <functional>

#include "EmbreeDevice.hpp"
#include "EmbreeGeometry.hpp"

namespace rmagine
{

/**
 * @brief EmbreeInstance
 * 
 * N instances belongs to 1 scene
 * M instances of 1 mesh
 * 
 * N instance belongs to 1 parent scene
 * 1 instance has one child scene
 * 
 */
class EmbreeInstance
: public EmbreeGeometry
{
public:
    using Base = EmbreeGeometry;

    EmbreeInstance(EmbreeDevicePtr device = embree_default_device() );
    virtual ~EmbreeInstance();

    void set(EmbreeScenePtr scene);
    EmbreeScenePtr scene();

    // Make this more comfortable to use
    // - functions as: setMesh(), or addMesh() ?
    // - translate rotate scale? 

    /**
     * @brief Apply transformation. Need to commit afterwards
     * 
     */
    void apply();

    virtual EmbreeGeometryType type() const
    {
        return EmbreeGeometryType::INSTANCE;
    }

private:

    // scene that is instanced by this object
    EmbreeScenePtr m_scene;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_INSTANCE_HPP