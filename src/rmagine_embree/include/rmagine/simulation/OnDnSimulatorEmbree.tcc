#include "OnDnSimulatorEmbree.hpp"
#include <rmagine/simulation/SimulationResults.hpp>
#include <limits>

#include "embree_common.h"

// TODO:
// RMAGINE_EMBREE_VERSION_MAJOR define is required
// since this is a template: Every program that uses 
// rmagine also require this definition. How to solve
// this problem without getting rid of templates?

namespace rmagine
{

template<typename BundleT>
void OnDnSimulatorEmbree::simulate(
    const Transform& Tbm,
    BundleT& ret) const
{
    MemoryView<Transform, RAM> Tbm_mem(&Tbm, 1);
    // TODO: change parallelization scheme for single simulations?
    simulate(Tbm_mem, ret);
}

template<typename BundleT>
BundleT OnDnSimulatorEmbree::simulate(
    const Transform& Tbm) const
{
    BundleT res;
    resize_memory_bundle<RAM>(res, m_model->getWidth(), m_model->getHeight(), 1);
    simulate(Tbm, res);
    return res;
}

template<typename BundleT>
void OnDnSimulatorEmbree::simulate(
    const MemoryView<Transform, RAM>& Tbm,
    BundleT& ret) const
{
    SimulationFlags flags = SimulationFlags::Zero();
    set_simulation_flags_<RAM>(ret, flags);
    
    const float range_min = m_model->range.min;
    const float range_max = m_model->range.max;

    #pragma omp parallel for
    for(size_t pid = 0; pid < Tbm.size(); pid++)
    {
        const Transform Tbm_ = Tbm[pid];
        const Transform Tsm_ = Tbm_ * m_Tsb[0];

        // TODO: only required for certain elements (Normals, ...)
        const Transform Tms_ = Tsm_.inv();

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
                rayhit.ray.tfar = range_max;
                rayhit.ray.mask = -1;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(m_map->scene->handle(), &rayhit);
                
                if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {
                    if constexpr(BundleT::template has<Hits<RAM> >())
                    {
                        if(flags.hits)
                        {
                            if(rayhit.ray.tfar >= range_min)
                            {
                                ret.Hits<RAM>::hits[glob_id] = 1;
                            } else {
                                ret.Hits<RAM>::hits[glob_id] = 0;
                            }
                        }
                    }

                    if constexpr(BundleT::template has<Ranges<RAM> >())
                    {
                        if(flags.ranges)
                        {
                            ret.Ranges<RAM>::ranges[glob_id] = rayhit.ray.tfar;
                        }
                    }

                    if constexpr(BundleT::template has<Points<RAM> >())
                    {
                        if(flags.points)
                        {
                            Vector pint = ray_dir_s * rayhit.ray.tfar;
                            ret.Points<RAM>::points[glob_id] = pint;
                        }
                    }

                    if constexpr(BundleT::template has<Normals<RAM> >())
                    {
                        if(flags.normals)
                        {
                            Vector nint{
                                    rayhit.hit.Ng_x,
                                    rayhit.hit.Ng_y,
                                    rayhit.hit.Ng_z
                                };
                            
                            nint.normalizeInplace();
                            nint = Tms_.R * nint;

                            // flip?
                            if(ray_dir_s.dot(nint) > 0.0)
                            {
                                nint *= -1.0;
                            }

                            // nint in local frame
                            ret.Normals<RAM>::normals[glob_id] = nint.normalize();
                        }
                    }

                    if constexpr(BundleT::template has<FaceIds<RAM> >())
                    {
                        if(flags.face_ids)
                        {
                            ret.FaceIds<RAM>::face_ids[glob_id] = rayhit.hit.primID;
                        }
                    }

                    if constexpr(BundleT::template has<GeomIds<RAM> >())
                    {
                        if(flags.geom_ids)
                        {
                            ret.GeomIds<RAM>::geom_ids[glob_id] = rayhit.hit.geomID;
                        }
                    }

                    if constexpr(BundleT::template has<ObjectIds<RAM> >())
                    {
                        if(flags.object_ids)
                        {
                            if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
                            {
                                ret.ObjectIds<RAM>::object_ids[glob_id] = rayhit.hit.instID[0];
                            } else {
                                ret.ObjectIds<RAM>::object_ids[glob_id] = rayhit.hit.geomID;
                            }
                        }
                    }
                    
                } else {
                    if constexpr(BundleT::template has<Hits<RAM> >())
                    {
                        if(flags.hits)
                        {
                            ret.Hits<RAM>::hits[glob_id] = 0;
                        }
                    }

                    if constexpr(BundleT::template has<Ranges<RAM> >())
                    {
                        if(flags.ranges)
                        {
                            ret.Ranges<RAM>::ranges[glob_id] = m_model->range.max + 1.0;
                        }
                    }

                    if constexpr(BundleT::template has<Points<RAM> >())
                    {
                        if(flags.points)
                        {
                            ret.Points<RAM>::points[glob_id].x = std::numeric_limits<float>::quiet_NaN();
                            ret.Points<RAM>::points[glob_id].y = std::numeric_limits<float>::quiet_NaN();
                            ret.Points<RAM>::points[glob_id].z = std::numeric_limits<float>::quiet_NaN();
                        }
                    }

                    if constexpr(BundleT::template has<Normals<RAM> >())
                    {
                        if(flags.normals)
                        {
                            ret.Normals<RAM>::normals[glob_id].x = std::numeric_limits<float>::quiet_NaN();
                            ret.Normals<RAM>::normals[glob_id].y = std::numeric_limits<float>::quiet_NaN();
                            ret.Normals<RAM>::normals[glob_id].z = std::numeric_limits<float>::quiet_NaN();
                        }
                    }

                    if constexpr(BundleT::template has<FaceIds<RAM> >())
                    {
                        if(flags.face_ids)
                        {
                            ret.FaceIds<RAM>::face_ids[glob_id] = std::numeric_limits<unsigned int>::max();
                        }
                    }

                    if constexpr(BundleT::template has<GeomIds<RAM> >())
                    {
                        if(flags.geom_ids)
                        {
                            ret.GeomIds<RAM>::geom_ids[glob_id] = std::numeric_limits<unsigned int>::max();
                        }
                    }

                    if constexpr(BundleT::template has<ObjectIds<RAM> >())
                    {
                        if(flags.object_ids)
                        {
                            ret.ObjectIds<RAM>::object_ids[glob_id] = std::numeric_limits<unsigned int>::max();
                        }
                    }
                }
            }
        }
    }
}

template<typename BundleT>
BundleT OnDnSimulatorEmbree::simulate(
    const MemoryView<Transform, RAM>& Tbm) const
{
    BundleT res;
    resize_memory_bundle<RAM>(res, m_model->getWidth(), m_model->getHeight(), Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine