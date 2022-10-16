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
 * @brief OptixGeometry
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_GEOMETRY_HPP
#define RMAGINE_MAP_OPTIX_GEOMETRY_HPP

#include <memory>
#include <rmagine/math/types.h>
#include <rmagine/util/optix/OptixContext.hpp>

#include <unordered_set>


#include "optix_definitions.h"
#include "OptixEntity.hpp"
#include "OptixTransformable.hpp"

namespace rmagine
{

class OptixGeometry
: public OptixEntity
, public OptixTransformable
{
public:
    OptixGeometry(OptixContextPtr context = optix_default_context());

    virtual ~OptixGeometry();

    // virtual void commit() = 0;

    virtual OptixGeometryType type() const = 0;

    // if child -> 0, else max of child + 1
    virtual unsigned int depth() const = 0;

    // virtual void apply() = 0 in OptixTransformable
    virtual void commit() = 0;

    // handle parents
    void cleanupParents();
    std::unordered_set<OptixScenePtr> parents() const;
    bool removeParent(OptixScenePtr parent);
    bool hasParent(OptixScenePtr parent) const;
    void addParent(OptixScenePtr parent);

    OptixScenePtr makeScene();
    OptixInstPtr instantiate();

protected:
    std::unordered_set<OptixSceneWPtr> m_parents;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_GEOMETRY_HPP

