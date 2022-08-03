#include "rmagine/map/embree/EmbreeDevice.hpp"

#include <iostream>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace rmagine {

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
    std::stringstream ss;
    ss << "[EmbreeDevice] Embree Error " << error << ": " << str;
    std::cout << ss.str() << std::endl; 
    // printf("[EmbreeDevice] Embree Error %d: %s\n", error, str);
    throw std::runtime_error(ss.str());

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
    // std::cout << "[EmbreeDevice::EmbreeDevice()] constructed." << std::endl;
}

EmbreeDevice::~EmbreeDevice()
{
    rtcReleaseDevice(m_device);
    // std::cout << "[EmbreeDevice::~EmbreeDevice()] destroyed." << std::endl;
}

/////////////////
RTCDevice EmbreeDevice::handle()
{
    return m_device;
}

EmbreeDevicePtr em_def_dev(new EmbreeDevice);

EmbreeDevicePtr embree_default_device()
{
    return em_def_dev;
}

void embree_default_device_reset()
{
    em_def_dev.reset();
}

} // namespace mamcl