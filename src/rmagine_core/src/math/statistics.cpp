#include "rmagine/math/statistics.h"
// #include "rmagine/math/linalg.h"
#include "rmagine/types/Memory.hpp"
#include <assert.h>

#include "rmagine/math/math.h"
#include "rmagine/math/omp.h"

#include <rmagine/util/prints.h>

#include <numeric>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>


namespace rmagine
{

RMAGINE_HOST_FUNCTION
void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats)
{
  stats[0] = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, dataset.points.size(), 128),
    CrossStatistics::Identity(),
    [&](const tbb::blocked_range<size_t>& r, CrossStatistics acc) 
    {
      for (size_t i = r.begin(); i != r.end(); ++i) 
      {
        if(  (dataset.mask.empty() || dataset.mask[i] > 0)
          && (model.mask.empty()   || model.mask[i]   > 0)
          && (dataset.ids.empty()  || dataset.ids[i] == params.dataset_id)
          && (model.ids.empty()    || model.ids[i]   == params.model_id)
          )
        {
          const Vector Di = pre_transform * dataset.points[i]; // read
          const Vector Mi = model.points[i]; // read

          // point to plane (P2P)
          const float dist = (Mi - Di).l2norm();

          if(fabs(dist) < params.max_dist)
          {
            // add Di -> Mi correspondence
            acc += CrossStatistics::Init(Di, Mi);
          }
        }
      }
      return acc;
    },
    std::plus<CrossStatistics>()
  );
}

RMAGINE_HOST_FUNCTION
void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
    MemoryView<CrossStatistics> stats_view(&stats, 1);
    statistics_p2p(pre_transform, dataset, model, params, stats_view);
}

RMAGINE_HOST_FUNCTION
CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params)
{
    CrossStatistics ret;
    statistics_p2p(pre_transform, dataset, model, params, ret);
    return ret;
}


RMAGINE_HOST_FUNCTION
void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats)
{
  stats[0] = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, dataset.points.size(), 128),
    CrossStatistics::Identity(),
    [&](const tbb::blocked_range<size_t>& r, CrossStatistics acc) 
    {
      for (size_t i = r.begin(); i != r.end(); ++i) 
      {
        if(    (dataset.mask.empty() || dataset.mask[i] > 0)
          && (model.mask.empty()   || model.mask[i]   > 0)
          && (dataset.ids.empty()  || dataset.ids[i] == params.dataset_id)
          && (model.ids.empty()    || model.ids[i]   == params.model_id)
          )
        {
          const Vector Di = pre_transform * dataset.points[i]; // read
          const Vector Ii = model.points[i]; // read
          const Vector Ni = model.normals[i];

          // 2. project new dataset point on plane -> new model point
          const float signed_plane_dist = (Ii - Di).dot(Ni);

          if(fabs(signed_plane_dist) < params.max_dist)
          {
            // nearest point on model
            const Vector Mi = Di + Ni * signed_plane_dist;

            // add Di -> Mi correspondence
            acc += CrossStatistics::Init(Di, Mi);
          }
        }
      }
      return acc;
    },
    std::plus<CrossStatistics>()
  );
}

RMAGINE_HOST_FUNCTION
void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
  MemoryView<CrossStatistics> stats_view(&stats, 1);
  statistics_p2l(pre_transform, dataset, model, params, stats_view);
}


RMAGINE_HOST_FUNCTION
CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params)
{
  CrossStatistics ret;
  statistics_p2l(pre_transform, dataset, model, params, ret);
  return ret;
}

RMAGINE_HOST_FUNCTION
void statistics_p2l_ow(
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const MemoryView<Transform>& model_pretransforms, // we would need more pre_transforms here
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats)
{
  // TODO: test this function!
  unsigned int n_measurements = dataset.points.size();
  unsigned int n_objects = stats.size();

  // initialize all partitions
  for(size_t i=0; i<n_objects; i++)
  {
    stats[i] = CrossStatistics::Identity();
  }

  for(size_t i=0; i<n_measurements; i++)
  {
    if(    (dataset.mask.empty() || dataset.mask[i] > 0)
        && (model.mask.empty()   || model.mask[i]   > 0)
        )
    {
      const unsigned int Oi = model.ids[i];
      const Transform Tmodel = model_pretransforms[Oi];

      // 1. read a few things we need to make a iterative
      const Vector Di   = dataset.points[i];
      const Vector Ii   = Tmodel * model.points[i];
      const Vector Ni   = Tmodel.R * model.normals[i];

      // 2. project new dataset point on plane -> new model point
      const float signed_plane_dist = (Ii - Di).dot(Ni);
      if(fabs(signed_plane_dist) < params.max_dist)
      {
        // nearest point on model
        const Vector Mi = Di + Ni * signed_plane_dist;
        // Or Mi -> Di here? since our model is going to matched to our dataset - reversed to localization
        stats[Oi] += CrossStatistics::Init(Di, Mi);
      }
    }
  }
}

// RMAGINE_HOST_FUNCTION
// std::unordered_map<unsigned int, CrossStatistics> statistics_p2l_ow(
//     const Transform& pre_transform,
//     const PointCloudView_<RAM>& dataset,
//     const PointCloudView_<RAM>& model,
//     const UmeyamaReductionConstraints params)
// {
//     std::unordered_map<unsigned int, CrossStatistics> ret;

//     // TODO
//     throw std::runtime_error("TODO");

//     return ret;
// }


} // namespace rmagine 