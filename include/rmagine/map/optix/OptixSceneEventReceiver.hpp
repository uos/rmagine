#ifndef RMAGINE_MAP_OPTIX_SCENE_EVENT_RECEIVER_HPP
#define RMAGINE_MAP_OPTIX_SCENE_EVENT_RECEIVER_HPP

#include "optix_definitions.h"

namespace rmagine
{

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