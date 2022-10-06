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
 * @brief EmbreeGeometry
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_GEOMETRY_HPP
#define RMAGINE_MAP_EMBREE_GEOMETRY_HPP

#include <memory>

#include <embree3/rtcore.h>

#include "EmbreeDevice.hpp"
#include "embree_definitions.h"

#include <rmagine/math/types.h>
#include <rmagine/math/linalg.h>

namespace rmagine
{

class EmbreeGeometry
: public std::enable_shared_from_this<EmbreeGeometry>
{
public:
    EmbreeGeometry(EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreeGeometry();

    // embree fields
    void setQuality(RTCBuildQuality quality);
    RTCGeometry handle() const;

    void setTransform(const Transform& T);
    Transform transform() const;

    /**
     * @brief Set the Transform object. matrix must not contain scale.
     * Otherwise call setTransformAndScale
     * 
     * @param T 
     */
    void setTransform(const Matrix4x4& T);
    void setTransformAndScale(const Matrix4x4& T);

    void setScale(const Vector3& S);
    Vector3 scale() const;

    /**
     * @brief Obtain composed matrix
     * 
     * @return Matrix4x4 
     */
    Matrix4x4 matrix() const;

    /**
     * @brief Apply all transformation changes to data if required
     * 
     */
    virtual void apply() {};

    void disable();
    
    void enable();

    void release();

    virtual void commit();

    virtual EmbreeGeometryType type() const = 0;

    EmbreeScenePtr makeScene();

    EmbreeInstancePtr instantiate();

    void cleanupParents();
    

    std::unordered_map<EmbreeSceneWPtr, unsigned int> ids();
    std::unordered_map<EmbreeSceneWPtr, unsigned int> ids() const;

    /**
     * @brief Get unique (per scene) ID
     * 
     * @param scene  scene the object was attached to
     * @return unsigned int  returns geometry id
     */
    unsigned int id(EmbreeScenePtr scene) const;

    std::unordered_set<EmbreeSceneWPtr> parents;
    std::string name;


protected:

    bool anyParentCommittedOnce() const;

    EmbreeDevicePtr m_device;
    RTCGeometry m_handle;

    Transform m_T;
    Vector3 m_S;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_GEOMETRY_HPP