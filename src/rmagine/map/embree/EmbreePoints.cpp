#include "rmagine/map/embree/EmbreePoints.hpp"

#include <iostream>

#include <map>
#include <cassert>

#include <rmagine/map/EmbreeDevice.hpp>
#include <rmagine/map/EmbreeScene.hpp>

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

void EmbreePoints::init(unsigned int Npoints)
{
    this->Npoints = Npoints;
    points = (PointWithRadius*)rtcSetNewGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(PointWithRadius), Npoints);
}

} // namespace rmagine