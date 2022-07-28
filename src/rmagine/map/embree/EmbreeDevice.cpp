#include "rmagine/map/embree/EmbreeDevice.hpp"

#include <iostream>
#include <cassert>

namespace rmagine {

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

/////////////////
// EmbreeDevice
/////////////////
EmbreeDevice::EmbreeDevice()
{
    m_device = rtcNewDevice(NULL);

    if (!m_device)
    {
        std::cerr << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;
    }

    rtcSetDeviceErrorFunction(m_device, errorFunction, NULL);
    std::cout << "[EmbreeDevice::~EmbreeDevice()] constructed." << std::endl;
}

EmbreeDevice::~EmbreeDevice()
{
    rtcReleaseDevice(m_device);
    std::cout << "[EmbreeDevice::~EmbreeDevice()] destroyed." << std::endl;
}

/////////////////
RTCDevice EmbreeDevice::handle()
{
    return m_device;
}

EmbreeDevicePtr bla(new EmbreeDevice);

EmbreeDevicePtr embree_default_device()
{
    return bla;
}

void embree_default_device_reset()
{
    bla.reset();
}

} // namespace mamcl