#include "rmagine/map/embree/EmbreePoints.hpp"

// other internal deps
#include "rmagine/map/embree/EmbreeDevice.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"

#include <iostream>

#include <map>
#include <cassert>



#include <rmagine/math/assimp_conversions.h>

namespace rmagine {

EmbreePoints::EmbreePoints(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_SPHERE_POINT);
}

EmbreePoints::EmbreePoints(unsigned int Npoints, EmbreeDevicePtr device)
:EmbreePoints(device)
{
    init(Npoints);
}

EmbreePoints::~EmbreePoints()
{

}

void EmbreePoints::init(unsigned int n_points)
{
    m_num_points = n_points;

    points.resize(n_points);

    // map to embree
    rtcSetSharedGeometryBuffer(m_handle,
                            RTC_BUFFER_TYPE_VERTEX,
                            0, // slot
                            RTC_FORMAT_FLOAT4, // RTCFormat
                            static_cast<const void*>(points.raw()), // ptr
                            0, // byteOffset
                            sizeof(PointWithRadius), // byteStride
                            m_num_points // itemCount
                            );
}

} // namespace rmagine