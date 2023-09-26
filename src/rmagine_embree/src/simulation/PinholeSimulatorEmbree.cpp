#include "rmagine/simulation/PinholeSimulatorEmbree.hpp"
#include <limits>

namespace rmagine
{

PinholeSimulatorEmbree::PinholeSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
    #if RMAGINE_EMBREE_VERSION_MAJOR == 3
    rtcInitIntersectContext(&m_context);
    #endif
}

PinholeSimulatorEmbree::PinholeSimulatorEmbree(const EmbreeMapPtr map)
:PinholeSimulatorEmbree()
{
    setMap(map);
}

PinholeSimulatorEmbree::~PinholeSimulatorEmbree()
{
    
}

void PinholeSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    m_map = map;
}

void PinholeSimulatorEmbree::setTsb(const MemoryView<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void PinholeSimulatorEmbree::setTsb(const Transform& Tsb)
{
    m_Tsb.resize(1);
    m_Tsb[0] = Tsb;
}

void PinholeSimulatorEmbree::setModel(const MemoryView<PinholeModel, RAM>& model)
{
    m_model = model;
}

void PinholeSimulatorEmbree::setModel(const PinholeModel& model)
{
    m_model.resize(1);
    m_model[0] = model;
}

void PinholeSimulatorEmbree::simulateRanges(
    const MemoryView<Transform, RAM>& Tbm,
    MemoryView<float, RAM>& ranges)
{
    #pragma omp parallel for
    for(size_t pid = 0; pid < Tbm.size(); pid++)
    {
        const Transform Tbm_ = Tbm[pid];
        const Transform Tsm_ = Tbm_ * m_Tsb[0];

        const unsigned int glob_shift = pid * m_model->size();

        for(unsigned int vid = 0; vid < m_model->getHeight(); vid++)
        {
            for(unsigned int hid = 0; hid < m_model->getWidth(); hid++)
            {
                const unsigned int loc_id = m_model->getBufferId(vid, hid);
                const unsigned int glob_id = glob_shift + loc_id;

                const Vector ray_dir_s = m_model->getDirection(vid, hid);
                const Vector ray_dir_m = Tsm_.R * ray_dir_s;

                RTCRayHit rayhit;
                rayhit.ray.org_x = Tsm_.t.x;
                rayhit.ray.org_y = Tsm_.t.y;
                rayhit.ray.org_z = Tsm_.t.z;
                rayhit.ray.dir_x = ray_dir_m.x;
                rayhit.ray.dir_y = ray_dir_m.y;
                rayhit.ray.dir_z = ray_dir_m.z;
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = std::numeric_limits<float>::infinity();
                rayhit.ray.mask = -1;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                #if RMAGINE_EMBREE_VERSION_MAJOR == 3
                rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);
                #elif RMAGINE_EMBREE_VERSION_MAJOR == 4
                rtcIntersect1(m_map->scene->handle(), &rayhit);
                #endif

                if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {
                    ranges[glob_id] = rayhit.ray.tfar;
                } else {
                    ranges[glob_id] = m_model->range.max + 1.0;
                }
            }
        }
    }
}

Memory<float, RAM> PinholeSimulatorEmbree::simulateRanges(
    const MemoryView<Transform, RAM>& Tbm)
{
    Memory<float, RAM> res(m_model->size() * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void PinholeSimulatorEmbree::simulateHits(
    const MemoryView<Transform, RAM>& Tbm, 
    MemoryView<uint8_t, RAM>& hits)
{
    #pragma omp parallel for
    for(size_t pid = 0; pid < Tbm.size(); pid++)
    {
        const Transform Tbm_ = Tbm[pid];
        const Transform Tsm_ = Tbm_ * m_Tsb[0];

        const unsigned int glob_shift = pid * m_model->size();

        for(unsigned int vid = 0; vid < m_model->getHeight(); vid++)
        {
            for(unsigned int hid = 0; hid < m_model->getWidth(); hid++)
            {
                const unsigned int loc_id = m_model->getBufferId(vid, hid);
                const unsigned int glob_id = glob_shift + loc_id;

                const Vector ray_dir_s = m_model->getDirection(vid, hid);
                const Vector ray_dir_m = Tsm_.R * ray_dir_s;

                RTCRayHit rayhit;
                rayhit.ray.org_x = Tsm_.t.x;
                rayhit.ray.org_y = Tsm_.t.y;
                rayhit.ray.org_z = Tsm_.t.z;
                rayhit.ray.dir_x = ray_dir_m.x;
                rayhit.ray.dir_y = ray_dir_m.y;
                rayhit.ray.dir_z = ray_dir_m.z;
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = std::numeric_limits<float>::infinity();
                rayhit.ray.mask = -1;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                #if RMAGINE_EMBREE_VERSION_MAJOR == 3
                rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);
                #elif RMAGINE_EMBREE_VERSION_MAJOR == 4
                rtcIntersect1(m_map->scene->handle(), &rayhit);
                #endif

                if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {
                    hits[glob_id] = 1;
                } else {
                    hits[glob_id] = 0;
                }
            }
        }
    }
}

Memory<uint8_t, RAM> PinholeSimulatorEmbree::simulateHits(
    const MemoryView<Transform, RAM>& Tbm)
{
    Memory<uint8_t, RAM> res(m_model->size() * Tbm.size());
    simulateHits(Tbm, res);
    return res;
}

} // namespace rmagine