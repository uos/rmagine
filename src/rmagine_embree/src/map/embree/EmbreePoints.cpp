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
    m_points.resize(n_points);
    m_points_transformed.resize(n_points);

    // map to embree
    rtcSetSharedGeometryBuffer(m_handle,
                            RTC_BUFFER_TYPE_VERTEX,
                            0, // slot
                            RTC_FORMAT_FLOAT4, // RTCFormat
                            static_cast<const void*>(m_points_transformed.raw()), // ptr
                            0, // byteOffset
                            sizeof(PointWithRadius), // byteStride
                            m_num_points // itemCount
                            );
}

void EmbreePoints::apply()
{
    for(size_t i=0; i<m_points.size(); i++)
    {
        m_points_transformed[i].p = m_T * m_points[i].p;
        m_points_transformed[i].r = m_points[i].r;
    }
    if(anyParentCommittedOnce())
    {
        rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
    }
}

MemoryView<PointWithRadius, RAM> EmbreePoints::points() const
{
    return m_points;
}

MemoryView<const PointWithRadius, RAM> EmbreePoints::pointsTransformed() const
{
    return MemoryView<const PointWithRadius, RAM>(m_points.raw(), m_num_points);
}

} // namespace rmagine