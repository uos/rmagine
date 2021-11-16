#include "imagine/simulation/EmbreeSimulator.hpp"

namespace imagine
{

EmbreeSimulator::EmbreeSimulator(const EmbreeMapPtr map)
:m_map(map)
{
    rtcInitIntersectContext(&m_context);
}

EmbreeSimulator::~EmbreeSimulator()
{
    
}

void EmbreeSimulator::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void EmbreeSimulator::setModel(const Memory<LiDARModel, RAM>& model)
{
    m_model = model;
}

void EmbreeSimulator::simulateRanges(
    const Memory<Transform, RAM>& Tbm,
    Memory<float, RAM>& ranges)
{
    #pragma omp parallel for
    for(size_t pid = 0; pid < Tbm.size(); pid++)
    {
        const Transform Tbm_ = Tbm[pid];
        const Transform Tsm_ = Tbm_ * m_Tsb[0];

        for(unsigned int vid = 0; vid < m_model->phi.size; vid++)
        {
            for(unsigned int hid = 0; hid < m_model->theta.size; hid++)
            {
                const unsigned int loc_id = vid * m_model->theta.size + hid;
                const unsigned int glob_id = pid * m_model->theta.size * m_model->phi.size + loc_id;

                const Vector ray_dir_s = m_model->getRay(vid, hid);
                const Vector ray_dir_m = Tsm_.R * ray_dir_s;

                RTCRayHit rayhit;
                rayhit.ray.org_x = Tsm_.t.x;
                rayhit.ray.org_y = Tsm_.t.y;
                rayhit.ray.org_z = Tsm_.t.z;
                rayhit.ray.dir_x = ray_dir_m.x;
                rayhit.ray.dir_y = ray_dir_m.y;
                rayhit.ray.dir_z = ray_dir_m.z;
                rayhit.ray.tnear = 0;
                rayhit.ray.tfar = INFINITY;
                rayhit.ray.mask = 0;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(m_map->scene, &m_context, &rayhit);

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

Memory<float, RAM> EmbreeSimulator::simulateRanges(
    const Memory<Transform, RAM>& Tbm)
{
    Memory<float, RAM> res(m_model->phi.size * m_model->theta.size * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

} // namespace imagine