#include "PinholeSimulatorEmbree.hpp"
#include <rmagine/simulation/SimulationResults.hpp>
#include <limits>

#include "embree_common.h"

#include <embree4/rtcore.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range3d.h>

// TODO:
// RMAGINE_EMBREE_VERSION_MAJOR define is required
// since this is a template: Every program that uses 
// rmagine also require this definition. How to solve
// this problem without getting rid of templates?

namespace rmagine
{

template<typename BundleT>
void PinholeSimulatorEmbree::simulate(
    const Transform& Tbm,
    BundleT& ret) const
{
  // TODO: change parallelization scheme for single simulations?
  const MemoryView<const Transform, RAM> Tbm_mem(&Tbm, 1);
  simulate(Tbm_mem, ret);
}

template<typename BundleT>
BundleT PinholeSimulatorEmbree::simulate(
    const Transform& Tbm) const
{
  BundleT res;
  resize_memory_bundle<RAM>(res, m_model->getWidth(), m_model->getHeight(), 1);
  simulate(Tbm, res);
  return res;
}

template<typename BundleT>
void PinholeSimulatorEmbree::simulate(
    const MemoryView<Transform, RAM>& Tbm,
    BundleT& ret) const
{
  const MemoryView<const Transform, RAM> Tbm_const(Tbm.raw(), Tbm.size());
  simulate(Tbm_const, ret);
}

template<typename BundleT>
void PinholeSimulatorEmbree::simulate(
    const MemoryView<const Transform, RAM>& Tbm,
    BundleT& ret) const
{
  SimulationFlags flags = SimulationFlags::Zero();
  set_simulation_flags_<RAM>(ret, flags);

  const float range_min = m_model->range.min;
  const float range_max = m_model->range.max;

  tbb::parallel_for( tbb::blocked_range3d<unsigned int>(
    0, Tbm.size(), 
    0, m_model->getHeight(), 
    0, m_model->getWidth()), 
    [&](const tbb::blocked_range3d<unsigned int>& r) 
  {
    for(unsigned int pid = r.pages().begin(), pid_end = r.pages().end(); pid < pid_end; pid++)
    {
      const Transform Tbm_ = Tbm[pid];
      const Transform Tsm_ = Tbm_ * m_Tsb[0];

      // TODO: only required for certain elements (Normals, ...)
      const Transform Tms_ = Tsm_.inv();

      const unsigned int glob_shift = pid * m_model->size();

      for(unsigned int vid = r.rows().begin(), vid_end = r.rows().end(); vid<vid_end; vid++)
      {
        for(unsigned int hid = r.cols().begin(), hid_end = r.cols().end(); hid<hid_end; hid++)
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
          rayhit.ray.tnear = 0; // if set to m_model->range.min we would scan through near occlusions
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
                // nint in map frame
                nint.normalizeInplace();
                nint = Tms_.R * nint;
                // nint in sensor frame

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
  });
}

template<typename BundleT>
BundleT PinholeSimulatorEmbree::simulate(
    const MemoryView<Transform, RAM>& Tbm) const
{
  BundleT res;
  resize_memory_bundle<RAM>(res, m_model->getWidth(), m_model->getHeight(), Tbm.size());
  simulate(Tbm, res);
  return res;
}

} // namespace rmagine