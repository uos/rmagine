#include "rmagine/simulation/O1DnSimulatorEmbree.hpp"

namespace rmagine
{

O1DnSimulatorEmbree::O1DnSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
    rtcInitIntersectContext(&m_context);
}

O1DnSimulatorEmbree::O1DnSimulatorEmbree(EmbreeMapPtr map)
:O1DnSimulatorEmbree()
{
    setMap(map);
}

O1DnSimulatorEmbree::~O1DnSimulatorEmbree()
{
    // std::cout << "O1DnSimulatorEmbree - Destructor " << std::endl;
}

void O1DnSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    m_map = map;
}

void O1DnSimulatorEmbree::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void O1DnSimulatorEmbree::setTsb(const Transform& Tsb)
{
    m_Tsb[0] = Tsb;
}

void O1DnSimulatorEmbree::setModel(const O1DnModel_<RAM>& model)
{
    m_model[0] = model;
}

void O1DnSimulatorEmbree::setModel(const Memory<O1DnModel_<RAM>, RAM>& model)
{
    m_model->width = model->width;
    m_model->height = model->height;
    m_model->range = model->range;
    m_model->orig = model->orig;
    m_model->dirs = model->dirs;

    // this is shallow copy becauce pointer is copied
    // m_model = model;
    // copy(model[0].dirs, m_model[0].dirs);
    // std::cout << "Set Model" << std::endl;
    // std::cout << model[0].dirs.raw() << " -> " << m_model[0].dirs.raw() << std::endl; 
}

void O1DnSimulatorEmbree::simulateRanges(
    const Memory<Transform, RAM>& Tbm,
    Memory<float, RAM>& ranges)
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

                const Vector ray_orig_s = m_model->getOrigin(vid, hid);
                const Vector ray_orig_m = Tsm_ * ray_orig_s;

                // std::cout << ray_dir_s.x << " " << ray_dir_s.y << " " << ray_dir_s.z << std::endl;

                RTCRayHit rayhit;
                rayhit.ray.org_x = ray_orig_m.x;
                rayhit.ray.org_y = ray_orig_m.y;
                rayhit.ray.org_z = ray_orig_m.z;
                rayhit.ray.dir_x = ray_dir_m.x;
                rayhit.ray.dir_y = ray_dir_m.y;
                rayhit.ray.dir_z = ray_dir_m.z;
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = INFINITY;
                rayhit.ray.mask = 0;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);

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

Memory<float, RAM> O1DnSimulatorEmbree::simulateRanges(
    const Memory<Transform, RAM>& Tbm)
{
    Memory<float, RAM> res(m_model->size() * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void O1DnSimulatorEmbree::simulateHits(
    const Memory<Transform, RAM>& Tbm, 
    Memory<uint8_t, RAM>& hits)
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

                const Vector ray_orig_s = m_model->getOrigin(vid, hid);
                const Vector ray_orig_m = Tsm_ * ray_orig_s;

                RTCRayHit rayhit;
                rayhit.ray.org_x = ray_orig_m.x;
                rayhit.ray.org_y = ray_orig_m.y;
                rayhit.ray.org_z = ray_orig_m.z;
                rayhit.ray.dir_x = ray_dir_m.x;
                rayhit.ray.dir_y = ray_dir_m.y;
                rayhit.ray.dir_z = ray_dir_m.z;
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = INFINITY;
                rayhit.ray.mask = 0;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);

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

Memory<uint8_t, RAM> O1DnSimulatorEmbree::simulateHits(
    const Memory<Transform, RAM>& Tbm)
{
    Memory<uint8_t, RAM> res(m_model->size() * Tbm.size());
    simulateHits(Tbm, res);
    return res;
}

} // namespace rmagine