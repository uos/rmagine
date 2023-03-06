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
 * @brief EmbreeScene
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_SCENE_HPP
#define RMAGINE_MAP_EMBREE_SCENE_HPP

#include "embree_definitions.h"
#include "EmbreeDevice.hpp"

#if RMAGINE_EMBREE_VERSION_MAJOR == 3
#include <embree3/rtcore.h>
#elif RMAGINE_EMBREE_VERSION_MAJOR == 4
#include <embree4/rtcore.h>
#else // RMAGINE_EMBREE_VERSION_MAJOR
#pragma message("Wrong major version of Embree found: ", RMAGINE_EMBREE_VERSION_MAJOR)
#endif // RMAGINE_EMBREE_VERSION_MAJOR


#include <memory>
#include <unordered_map>
#include <optional>
#include <assimp/scene.h>

namespace rmagine
{
struct EmbreeSceneSettings
{
    /**
     * @brief quality
     * - RTC_BUILD_QUALITY_LOW
     * - RTC_BUILD_QUALITY_MEDIUM
     * - RTC_BUILD_QUALITY_HIGH
     * - RTC_BUILD_QUALITY_REFIT
     * 
     */
    RTCBuildQuality quality = RTCBuildQuality::RTC_BUILD_QUALITY_MEDIUM;
    
    /**
     * @brief flags can be combined by FLAGA | FLAGB ...
     * 
     * - RTC_SCENE_FLAG_NONE (default)
     * - RTC_SCENE_FLAG_DYNAMIC
     * - RTC_SCENE_FLAG_COMPACT
     * - RTC_SCENE_FLAG_ROBUST
     * - RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION
     */
    RTCSceneFlags flags = RTCSceneFlags::RTC_SCENE_FLAG_NONE;
};

/**
 * @brief EmbreeScene
 * 
 * - meshes
 * - instances
 * 
 */
class EmbreeScene
: public std::enable_shared_from_this<EmbreeScene>
{
public:
    EmbreeScene(
        EmbreeSceneSettings settings = {}, 
        EmbreeDevicePtr device = embree_default_device());
        
    ~EmbreeScene();

    void setQuality(RTCBuildQuality quality);

    void setFlags(RTCSceneFlags flags);

    unsigned int add(EmbreeGeometryPtr geom);
    std::optional<unsigned int> getOpt(const EmbreeGeometryPtr geom) const;
    std::optional<unsigned int> getOpt(const std::shared_ptr<const EmbreeGeometry> geom) const;
    
    unsigned int get(const EmbreeGeometryPtr geom) const;
    unsigned int get(const std::shared_ptr<const EmbreeGeometry> geom) const;

    bool has(const EmbreeGeometryPtr geom) const;
    bool has(const std::shared_ptr<const EmbreeGeometry> geom) const;
    bool remove(EmbreeGeometryPtr geom);

    EmbreeGeometryPtr get(const unsigned int geom_id) const;
    bool has(const unsigned int geom_id) const;
    EmbreeGeometryPtr remove(const unsigned int geom_id);

    template<typename T>
    std::shared_ptr<T> getAs(const unsigned int geom_id) const;

    std::unordered_map<EmbreeGeometryPtr, unsigned int> ids() const;
    std::unordered_map<unsigned int, EmbreeGeometryPtr> geometries() const;

    template<typename T>
    unsigned int count() const;

    RTCScene handle();

    void commit();

    EmbreeInstancePtr instantiate();

    /**
     * @brief find all leaf geometries recursively
     * 
     * @param scene 
     * @return std::unordered_set<EmbreeMeshPtr> 
     */
    std::unordered_set<EmbreeGeometryPtr> findLeafs() const;

    /**
     * @brief check if scene was committed at least one time.
     * important for functions as rtcUpdateGeometryBuffer:
     * "this function needs to be called only when doing buffer modifications after the first rtcCommitScene"
     * 
     * @return true 
     * @return false 
     */
    bool committedOnce() const;

    bool isTopLevel() const;

    std::unordered_map<unsigned int, unsigned int> integrate(EmbreeScenePtr other);

    /**
     * @brief freeze makes the map static. 
     * 
     * Improve ray traversal time -> decreases map update time
     * 
     * Steps:
     * - any geometry that has only one instance
     *    - instance destroyed
     *    - instance transform applied to geometry
     *    - transformed geometry added to root
     * 
     * Drawbacks:
     * - Recover to dynamic map not tested yet. use with care
     * 
     */
    void freeze();

    inline EmbreeDevicePtr device() const 
    {
        return m_device;
    }

    // Scene has no right to let parents stay alive
    std::unordered_set<EmbreeInstanceWPtr> parents;
private:

    std::unordered_map<unsigned int, EmbreeGeometryPtr > m_geometries;
    std::unordered_map<EmbreeGeometryPtr, unsigned int> m_ids;

    bool m_committed_once = false;

    RTCScene m_scene;
    EmbreeDevicePtr m_device;
};


template<typename T>
unsigned int EmbreeScene::count() const
{
    unsigned int ret = 0;

    for(auto it = m_geometries.begin(); it != m_geometries.end(); ++it)
    {
        if(std::dynamic_pointer_cast<T>(it->second))
        {
            ret++;
        }
    }

    return ret;
}


template<typename T>
std::shared_ptr<T> EmbreeScene::getAs(const unsigned int geom_id) const
{
    std::shared_ptr<T> ret;

    EmbreeGeometryPtr geom = get(geom_id);
    if(geom)
    {
        ret = std::dynamic_pointer_cast<T>(geom);
    }

    return ret;
}


EmbreeScenePtr make_embree_scene(
    const aiScene* ascene,
    EmbreeDevicePtr device = embree_default_device());


} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_SCENE_HPP