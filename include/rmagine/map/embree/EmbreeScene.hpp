#ifndef RMAGINE_MAP_EMBREE_SCENE_HPP
#define RMAGINE_MAP_EMBREE_SCENE_HPP

#include "embree_types.h"
#include "EmbreeDevice.hpp"

#include <embree3/rtcore.h>
#include <memory>
#include <unordered_map>
#include <optional>

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
     * @brief 
     * 
     */
    void optimize();

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


} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_SCENE_HPP