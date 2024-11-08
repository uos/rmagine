#include "rmagine/math/statistics.h"
// #include "rmagine/math/linalg.h"
#include "rmagine/types/Memory.hpp"
#include <assert.h>

#include "rmagine/math/math.h"
#include "rmagine/math/omp.h"

#include <rmagine/util/prints.h>

namespace rmagine
{

bool alwaysFalse()
{
    std::cout << "BLA!!!" << std::endl;
    return false;
}

RMAGINE_HOST_FUNCTION
void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
    stats = statistics_p2p(pre_transform, dataset, model, params);
}

RMAGINE_HOST_FUNCTION
CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params)
{
    // since all of our data is already there we can do a 2 stage mean reduction
    Vector3 dataset_mean = Vector3::Zeros();
    Vector3 model_mean = Vector3::Zeros();
    unsigned int n_meas = 0;

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
        return CrossStatistics::Identity();
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

    return {dataset_mean, model_mean, covariance, n_meas};
}

RMAGINE_HOST_FUNCTION
void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats)
{
    stats = statistics_p2l(pre_transform, dataset, model, params);
}


RMAGINE_HOST_FUNCTION
CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params)
{
    // since all of our data is already there we can do a 2 stage mean reduction
    Vector3 dataset_mean = Vector3::Zeros();
    Vector3 model_mean = Vector3::Zeros();
    unsigned int n_meas = 0;

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
        return CrossStatistics::Identity();
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

    return {dataset_mean, model_mean, covariance, n_meas};
}

} // namespace rmagine 