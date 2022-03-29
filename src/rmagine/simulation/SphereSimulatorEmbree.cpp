#include "rmagine/simulation/SphereSimulatorEmbree.hpp"

#include <rmagine/util/prints.h>

#include <rmagine/util/StopWatch.hpp>

namespace rmagine
{

SphereSimulatorEmbree::SphereSimulatorEmbree()
:m_model(1)
,m_Tsb(1)
{
    m_Tsb[0].setIdentity();
    // m_context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
    rtcInitIntersectContext(&m_context);
    
}

SphereSimulatorEmbree::SphereSimulatorEmbree(EmbreeMapPtr map)
:SphereSimulatorEmbree()
{
    setMap(map);
}

SphereSimulatorEmbree::~SphereSimulatorEmbree()
{
    
}

void SphereSimulatorEmbree::setMap(EmbreeMapPtr map)
{
    // std::cout << "[RMagine - SphereSimulatorEmbree] setMap" << std::endl;
    m_map = map;
}

void SphereSimulatorEmbree::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void SphereSimulatorEmbree::setTsb(const Transform& Tsb)
{
    // std::cout << "[RMagine - SphereSimulatorEmbree] setTsb" << std::endl;
    m_Tsb.resize(1);
    m_Tsb[0] = Tsb;
}

void SphereSimulatorEmbree::setModel(const Memory<SphericalModel, RAM>& model)
{
    m_model = model;
}

void SphereSimulatorEmbree::setModel(const SphericalModel& model)
{
    m_model.resize(1);
    m_model[0] = model;
}

void SphereSimulatorEmbree::simulateRanges(
    const Memory<Transform, RAM>& Tbm,
    Memory<float, RAM>& ranges)
{
    std::cout << "HALLO" << std::endl << std::endl;

    StopWatch sw;
    double el;

    el = sw();

    // #pragma omp parallel for
    // for(size_t pid = 0; pid < Tbm.size(); pid++)
    // {
    //     const Transform Tbm_ = Tbm[pid];
    //     const Transform Tsm_ = Tbm_ * m_Tsb[0];

    //     const unsigned int glob_shift = pid * m_model->size();

    //     for(unsigned int vid = 0; vid < m_model->getHeight(); vid++)
    //     {
    //         for(unsigned int hid = 0; hid < m_model->getWidth(); hid++)
    //         {
    //             const unsigned int loc_id = m_model->getBufferId(vid, hid);
    //             const unsigned int glob_id = glob_shift + loc_id;

    //             const Vector ray_dir_s = m_model->getDirection(vid, hid);
    //             const Vector ray_dir_m = Tsm_.R * ray_dir_s;

    //             RTCRayHit rayhit;
    //             rayhit.ray.org_x = Tsm_.t.x;
    //             rayhit.ray.org_y = Tsm_.t.y;
    //             rayhit.ray.org_z = Tsm_.t.z;
    //             rayhit.ray.dir_x = ray_dir_m.x;
    //             rayhit.ray.dir_y = ray_dir_m.y;
    //             rayhit.ray.dir_z = ray_dir_m.z;
    //             rayhit.ray.tnear = 0;
    //             rayhit.ray.tfar = INFINITY;
    //             rayhit.ray.mask = 0;
    //             rayhit.ray.flags = 0;
    //             rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    //             rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    //             rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);

    //             if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
    //             {    
    //                 ranges[glob_id] = rayhit.ray.tfar;
    //             } else {
    //                 ranges[glob_id] = m_model->range.max + 1.0;
    //             }
    //         }
    //     }
    // }

    el = sw();

    std::cout << "Old: " << Tbm.size() << " -> " << el << "s" << std::endl;

    el = sw();

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

                // RTCRayHit rayhit;
                // rayhit.ray.org_x = Tsm_.t.x;
                // rayhit.ray.org_y = Tsm_.t.y;
                // rayhit.ray.org_z = Tsm_.t.z;
                // rayhit.ray.dir_x = ray_dir_m.x;
                // rayhit.ray.dir_y = ray_dir_m.y;
                // rayhit.ray.dir_z = ray_dir_m.z;
                // rayhit.ray.tnear = 0;
                // rayhit.ray.tfar = INFINITY;
                // rayhit.ray.mask = 0;
                // rayhit.ray.flags = 0;
                // rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                // rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                // rtcIntersect1(m_map->scene->handle(), &m_context, &rayhit);

                // if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                // {    
                //     ranges[glob_id] = rayhit.ray.tfar;
                // } else {
                //     ranges[glob_id] = m_model->range.max + 1.0;
                // }
            }
        }
    }


    el = sw();

    std::cout << "Ray gen only: " << Tbm.size() << " -> " << el << "s" << std::endl;


    std::cout << "New: " << std::endl;
    std::vector<RTCRayHit> rayhits(Tbm.size() * m_model->size());
    // Memory<RTCRayHit, RAM> rayhits(Tbm.size() * m_model->size());

    el = sw();
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

                RTCRayHit& rayhit = rayhits[glob_id];
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
            }
        }
    }

    el = sw();

    std::cout << "Ray gen: " << Tbm.size() << " -> " << el << "s" << std::endl;


    sw();


    for(size_t pid = 0; pid < Tbm.size(); pid++)
    {
        const unsigned int glob_shift = pid * m_model->size();
        RTCRayHit* buffer = &rayhits[glob_shift];

        const unsigned int Nelements = m_model.size();
        rtcIntersect1M(m_map->scene->handle(), &m_context, buffer, Nelements, sizeof(RTCRayHit));
    }

    el = sw();

    std::cout << "Ray intersect: " << Tbm.size() << " -> " << el << "s" << std::endl;

    sw();

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

                const RTCRayHit& rayhit = rayhits[glob_id];

                if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {    
                    ranges[glob_id] = rayhit.ray.tfar;
                } else {
                    ranges[glob_id] = m_model->range.max + 1.0;
                }
            }
        }
    }


    el = sw();

    std::cout << "Reading results: " << Tbm.size() << " -> " << el << "s" << std::endl;

    std::cout << std::endl;
}

Memory<float, RAM> SphereSimulatorEmbree::simulateRanges(
    const Memory<Transform, RAM>& Tbm)
{
    Memory<float, RAM> res(m_model->phi.size * m_model->theta.size * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void SphereSimulatorEmbree::simulateHits(
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

Memory<uint8_t, RAM> SphereSimulatorEmbree::simulateHits(
    const Memory<Transform, RAM>& Tbm)
{
    Memory<uint8_t, RAM> res(m_model->phi.size * m_model->theta.size * Tbm.size());
    simulateHits(Tbm, res);
    return res;
}

} // namespace rmagine