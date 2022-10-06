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
 * @brief Contains a list of forward declared objectes
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_DEFINTIONS_HPP
#define RMAGINE_MAP_OPTIX_DEFINTIONS_HPP

#include <memory>
#include <rmagine/util/hashing.h>

namespace rmagine
{


enum class OptixGeometryType
{
    NONE,
    INSTANCE,
    MESH,
    POINTS
};

enum class OptixSceneType
{
    NONE,
    INSTANCES,
    GEOMETRIES
};


class OptixEntity;
class OptixTransformable;
class OptixAccelerationStructure;
class OptixGeometry;
class OptixInst;
class OptixScene;
struct OptixSceneCommitResult;
class OptixSceneEventReceiver;
struct OptixInstanceData;


using OptixEntityPtr = std::shared_ptr<OptixEntity>;
using OptixTransformablePtr = std::shared_ptr<OptixTransformable>;
using OptixAccelerationStructurePtr = std::shared_ptr<OptixAccelerationStructure>;
using OptixGeometryPtr = std::shared_ptr<OptixGeometry>;
using OptixInstPtr = std::shared_ptr<OptixInst>;
using OptixScenePtr = std::shared_ptr<OptixScene>;
using OptixSceneEventReceiverPtr = std::shared_ptr<OptixSceneEventReceiver>;

using OptixEntityWPtr = std::weak_ptr<OptixEntity>;
using OptixTransformableWPtr = std::weak_ptr<OptixTransformable>;
using OptixAccelerationStructureWPtr = std::weak_ptr<OptixAccelerationStructure>;
using OptixGeometryWPtr = std::weak_ptr<OptixGeometry>;
using OptixInstWPtr = std::weak_ptr<OptixInst>;
using OptixSceneWPtr = std::weak_ptr<OptixScene>;
using OptixSceneEventReceiverWPtr = std::weak_ptr<OptixSceneEventReceiver>;




} // namespace rmagine

namespace std
{

// INSTANCE
template<>
struct hash<rmagine::OptixInstWPtr> 
    : public rmagine::weak_hash<rmagine::OptixInst>
{};

template<>
struct equal_to<rmagine::OptixInstWPtr> 
    : public rmagine::weak_equal_to<rmagine::OptixInst>
{};

template<>
struct less<rmagine::OptixInstWPtr> 
    : public rmagine::weak_less<rmagine::OptixInst>
{};



// SCENE
template<>
struct hash<rmagine::OptixSceneWPtr> 
    : public rmagine::weak_hash<rmagine::OptixScene>
{};

template<>
struct equal_to<rmagine::OptixSceneWPtr> 
    : public rmagine::weak_equal_to<rmagine::OptixScene>
{};

template<>
struct less<rmagine::OptixSceneWPtr> 
    : public rmagine::weak_less<rmagine::OptixScene>
{};



// SCENE EVENT RECEIVER
template<>
struct hash<rmagine::OptixSceneEventReceiverWPtr> 
    : public rmagine::weak_hash<rmagine::OptixSceneEventReceiver>
{};

template<>
struct equal_to<rmagine::OptixSceneEventReceiverWPtr> 
    : public rmagine::weak_equal_to<rmagine::OptixSceneEventReceiver>
{};

template<>
struct less<rmagine::OptixSceneEventReceiverWPtr> 
    : public rmagine::weak_less<rmagine::OptixSceneEventReceiver>
{};


} // namespace std

#endif // RMAGINE_MAP_OPTIX_DEFINTIONS_HPP