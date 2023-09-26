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
 * @brief Contains of commonly used shapes for fast instancing
 * 
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_SHAPES_H
#define RMAGINE_MAP_EMBREE_SHAPES_H

#include "EmbreeMesh.hpp"
#include "EmbreeDevice.hpp"
#include <memory>

namespace rmagine
{


class EmbreeSphere;
using EmbreeSpherePtr = std::shared_ptr<EmbreeSphere>;
using EmbreeSphereWPtr = std::weak_ptr<EmbreeSphere>;

class EmbreeSphere : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreeSphere(
        unsigned int num_long = 50, 
        unsigned int num_lat = 50,
        EmbreeDevicePtr device = embree_default_device()
    );
};

class EmbreeCube;
using EmbreeCubePtr = std::shared_ptr<EmbreeCube>;
using EmbreeCubeWPtr = std::weak_ptr<EmbreeCube>;

class EmbreeCube : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreeCube(
        unsigned int side_triangles_exp = 1,
        EmbreeDevicePtr device = embree_default_device()
    );
};

class EmbreePlane;
using EmbreePlanePtr = std::shared_ptr<EmbreePlane>;
using EmbreePlaneWPtr = std::weak_ptr<EmbreePlane>;

class EmbreePlane : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreePlane(
        unsigned int side_triangles_exp = 1,
        EmbreeDevicePtr device = embree_default_device()
    );
};


class EmbreeCylinder;
using EmbreeCylinderPtr = std::shared_ptr<EmbreeCylinder>;
using EmbreeCylinderWPtr = std::weak_ptr<EmbreeCylinder>;

class EmbreeCylinder : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreeCylinder(
        unsigned int side_faces = 100,
        EmbreeDevicePtr device = embree_default_device()
    );
};



} // namespace rmagine


#endif // RMAGINE_MAP_EMBREE_SHAPES_H