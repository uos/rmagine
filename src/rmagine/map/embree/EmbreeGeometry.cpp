
#include "rmagine/map/embree/EmbreeGeometry.hpp"
#include <iostream>

namespace rmagine
{

EmbreeGeometry::EmbreeGeometry(EmbreeDevicePtr device)
:m_device(device)
{
    std::cout << "[EmbreeGeometry::EmbreeGeometry()] constructed." << std::endl;
}

EmbreeGeometry::~EmbreeGeometry()
{
    std::cout << "[EmbreeGeometry::~EmbreeGeometry()] destroyed." << std::endl;
}

RTCGeometry EmbreeGeometry::handle() const
{
    return m_handle;
}

void EmbreeGeometry::disable()
{
    rtcDisableGeometry(m_handle);
}

void EmbreeGeometry::enable()
{
    rtcEnableGeometry(m_handle);
}

void EmbreeGeometry::release()
{
    rtcReleaseGeometry(m_handle);
}

void EmbreeGeometry::commit()
{
    rtcCommitGeometry(m_handle);
}

} // namespace rmagine

