#ifndef RMAGINE_MAP_EMBREE_SCENE_HPP
#define RMAGINE_MAP_EMBREE_SCENE_HPP

#include "embree_types.h"
#include "EmbreeDevice.hpp"

#include <embree3/rtcore.h>
#include <memory>

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
     * @brief flags
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
 * @brief 
 * 
 */
class EmbreeScene
{
public:
    EmbreeScene(EmbreeDevicePtr device, EmbreeSceneSettings settings = {});
    ~EmbreeScene();

    void setQuality(RTCBuildQuality quality);

    void setFlags(RTCSceneFlags flags);

    void add(EmbreeInstancePtr inst);
    EmbreeInstanceSet instances() const;
    void remove(EmbreeInstancePtr inst);


    void add(EmbreeMeshPtr mesh);
    EmbreeInstanceSet meshes() const;
    void remove(EmbreeMeshPtr mesh);

    

    RTCScene handle();

    void commit();

private:
    EmbreeInstanceSet m_instances;
    EmbreeMeshSet m_meshes;

    RTCScene m_scene;
    EmbreeDevicePtr m_device;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_SCENE_HPP