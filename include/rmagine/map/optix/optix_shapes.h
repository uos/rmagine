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
 * @brief Contains a list of shapes for fast instantiation
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_SHAPES_H
#define RMAGINE_MAP_OPTIX_SHAPES_H

#include "optix_definitions.h"
#include "OptixMesh.hpp"

namespace rmagine
{

class OptixSphere : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixSphere(unsigned int num_long = 50,
        unsigned int num_lat = 50,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixSphere();
};

using OptixSpherePtr = std::shared_ptr<OptixSphere>;

class OptixCube : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixCube(unsigned int side_triangles_exp = 1,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixCube();
};

using OptixCubePtr = std::shared_ptr<OptixCube>;


class OptixPlane : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixPlane(unsigned int side_triangles_exp = 1,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixPlane();
};

using OptixPlanePtr = std::shared_ptr<OptixPlane>;
class OptixCylinder : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixCylinder(unsigned int side_faces = 100,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixCylinder();
};

using OptixCylinderPtr = std::shared_ptr<OptixCylinder>;


} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SHAPES_H