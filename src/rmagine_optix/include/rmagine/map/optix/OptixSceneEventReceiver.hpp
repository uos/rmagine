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
 * @brief OptixSceneEventReceiver
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_SCENE_EVENT_RECEIVER_HPP
#define RMAGINE_MAP_OPTIX_SCENE_EVENT_RECEIVER_HPP

#include "optix_definitions.h"

namespace rmagine
{

/**
 * @brief implement this class to receive events from a changing optix scene
 * 
 * Example:
 * 
 * @code{cpp}
 * class MyOptixSceneEventReceiver : public OptixSceneEventReceiver
 * {
 * public:
 *     virtual onCommitDone() override
 *     {
 *         std::cout << "Scene has changed!" << std::endl;
 *     }
 * };
 * @endcode
 * 
 * your OptixSceneEventReceiver you can then use like
 * 
 * @code{cpp}
 * 
 * OptixMapPtr gpu_map = import_optix_map(path_to_mesh);
 * 
 * auto rec = std::make_shared<MyOptixSceneEventReceiver>();
 * gpu_map->scene()->addEventReceiver(rec);
 * 
 * @endcode
 * 
 * The example code prints "Scene has changed!" on every commit of the scene.
 * 
 */
class OptixSceneEventReceiver
: std::enable_shared_from_this<OptixSceneEventReceiver>
{
protected:
    virtual ~OptixSceneEventReceiver();

    // 1.
    virtual void onDepthChanged() {};

    // 2.
    virtual void onSBTUpdated(bool size_changed) {};

    // 3.
    virtual void onCommitDone(
        const OptixSceneCommitResult& info) {};

    OptixSceneWPtr m_scene;
private:
    
    friend class OptixScene;
};

} // rmagine


#endif // RMAGINE_MAP_OPTIX_SCENE_EVENT_RECEIVER_HPP