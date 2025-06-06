#include "rmagine/math/statistics.h"
// #include "rmagine/math/linalg.h"
#include "rmagine/types/Memory.hpp"
#include <assert.h>

#include "rmagine/math/math.h"
#include "rmagine/math/omp.h"

#include <rmagine/util/prints.h>



#if PARALLEL
#include <execution>
#define PAR std::execution::par,
#else
#define PAR
#endif

#include <numeric>


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

    // since all of our data is already there we can do a 2 stage mean reduction
    Vector3 dataset_mean = Vector3::Zeros();
    Vector3 model_mean = Vector3::Zeros();
    unsigned int n_meas = 0;

    // TODO: test transform reduce
    // is this available with ubuntu 20?
    // auto sum3 = std::transform_reduce(PAR v.cbegin(), v.cend(), 0L, std::plus{},
    //                                   [](auto val) { return val * val; });

    #pragma omp parallel for reduction(+: dataset_mean, model_mean, n_meas)
    for(size_t i=0; i<dataset.points.size(); i++)
    {
        if(    (dataset.mask.empty() || dataset.mask[i] > 0)
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
                dataset_mean += Di;
                model_mean += Mi;
                n_meas += 1;
            }
        }
    }

    dataset_mean /= static_cast<double>(n_meas);
    model_mean /= static_cast<double>(n_meas);

    if(n_meas == 0)
    {
        stats[0] = CrossStatistics::Identity();
        return;
    }

    Matrix3x3 covariance = Matrix3x3::Zeros();

    #pragma omp parallel for reduction(+: covariance)
    for(size_t i=0; i<dataset.points.size(); i++)
    {
        if(    (dataset.mask.empty() || dataset.mask[i] > 0)
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
                covariance += (Mi - model_mean).multT(Di - dataset_mean);
            }
        }
    }

    covariance /= static_cast<double>(n_meas);

    // write
    stats[0].dataset_mean = dataset_mean;
    stats[0].model_mean = model_mean;
    stats[0].covariance = covariance;
    stats[0].n_meas = n_meas;
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
    // TODO: figure out what to do with openmp here
    // - it gets much faster for non-masked datasets
    // - it gets much slower for reductions with small masks, e.g. one skrewdriver on a scanned tabletop

    // TODO: I rather want to have the one-pass cov algorithm here. Also to have the same algorithms running everywhere
    // - for now it's not working, see function "statistics_p2l_iwantthis_butnans"


    // since all of our data is already there we can do a 2 stage mean reduction
    Vector3 dataset_mean = Vector3::Zeros();
    Vector3 model_mean = Vector3::Zeros();
    unsigned int n_meas = 0;

    // #pragma omp parallel for reduction(+: dataset_mean, model_mean, n_meas)
    for(size_t i=0; i<dataset.points.size(); i++)
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
                dataset_mean += Di;
                model_mean += Mi;
                n_meas += 1;
            }
        }
    }

    dataset_mean /= static_cast<double>(n_meas);
    model_mean /= static_cast<double>(n_meas);

    if(n_meas == 0)
    {
        stats[0] = CrossStatistics::Identity();
        return;
    }

    Matrix3x3 covariance = Matrix3x3::Zeros();

    // #pragma omp parallel for reduction(+: covariance)
    for(size_t i=0; i<dataset.points.size(); i++)
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
                covariance += (Mi - model_mean).multT(Di - dataset_mean);
            }
        }
    }

    covariance /= static_cast<double>(n_meas);

    // write
    stats[0].dataset_mean = dataset_mean;
    stats[0].model_mean = model_mean;
    stats[0].covariance = covariance;
    stats[0].n_meas = n_meas;
}


RMAGINE_HOST_FUNCTION
void statistics_p2l_iwantthis_butnans(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats)
{
    CrossStatistics st = CrossStatistics::Identity();
    
    // #pragma omp parallel for reduction(+: st)
    for(size_t i=0; i<dataset.points.size(); i++)
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
                st += CrossStatistics::Init(Di, Mi);
            }
        }
    }

    // write
    stats[0] = st;
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