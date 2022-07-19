#include "rmagine/map/EmbreeDevice.hpp"

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
}

EmbreeDevice::~EmbreeDevice()
{
    rtcReleaseDevice(m_device);
}

/////////////////
RTCDevice EmbreeDevice::handle()
{
    return m_device;
}

} // namespace mamcl