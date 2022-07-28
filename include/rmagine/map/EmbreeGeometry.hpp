#ifndef RMAGINE_MAP_EMBREE_GEOMETRY_HPP
#define RMAGINE_MAP_EMBREE_GEOMETRY_HPP

#include <memory>

#include <embree3/rtcore.h>

#include "EmbreeDevice.hpp"
#include "embree_types.h"

namespace rmagine
{

class EmbreeGeometry
: public std::enable_shared_from_this<EmbreeGeometry>
{
public:
    EmbreeGeometry(EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreeGeometry();

    // embree fields
    RTCGeometry handle() const;


    void disable();
    
    void enable();

    void release();

    virtual void commit();

    EmbreeSceneWPtr parent;
    unsigned int id;
    std::string name;

protected:
    EmbreeDevicePtr m_device;
    RTCGeometry m_handle;
};

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_GEOMETRY_HPP