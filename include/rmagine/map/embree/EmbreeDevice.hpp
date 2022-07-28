
#ifndef RMAGINE_MAP_EMBREE_DEVICE_HPP
#define RMAGINE_MAP_EMBREE_DEVICE_HPP

#include <embree3/rtcore.h>
#include <memory>

namespace rmagine
{

class EmbreeDevice
{
public:
    EmbreeDevice();

    ~EmbreeDevice();

    RTCDevice handle();

private:
    RTCDevice m_device;
};

using EmbreeDevicePtr = std::shared_ptr<EmbreeDevice>;

// static EmbreeDevicePtr embree_default_device(new EmbreeDevice);

EmbreeDevicePtr embree_default_device();

void embree_default_device_reset();

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_DEVICE_HPP