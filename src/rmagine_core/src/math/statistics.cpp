#include "rmagine/math/statistics.h"
// #include "rmagine/math/linalg.h"
#include "rmagine/types/Memory.hpp"
#include <assert.h>

#include "rmagine/math/math.h"
#include "rmagine/math/omp.h"

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
    CrossStatistics stats;

    #pragma omp parallel for shared(pre_transform, dataset, model, params) reduction(+: stats)
    for(size_t i=0; i<dataset.points.size(); i++)
    {
        // figure out if distance is too high
        if(    (dataset.mask.empty() || dataset.mask[i] > 0)
            && (model.mask.empty()   || model.mask[i]   > 0)
            && (dataset.ids.empty()  || dataset.ids[i] == params.dataset_id)
            && (model.ids.empty()    || model.ids[i]   == params.model_id)
            )
        {
            if(!dataset.ids.empty())
            {
                std::cout << "Hello! Dataset Id " << dataset.ids[i] << ", mask id: " << dataset.mask[i] << std::endl;
            }

            const Vector Di = pre_transform * dataset.points[i]; // read
            const Vector Mi = model.points[i]; // read

            // point to plane (P2P)
            const float dist = (Mi - Di).l2norm();

            if(fabs(dist) < params.max_dist)
            {
                // add Di -> Mi correspondence
                stats += CrossStatistics::Init(Di, Mi);
            }
        }
    }

    return stats;
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
    CrossStatistics stats = CrossStatistics::Identity();;

    #pragma omp parallel for shared(pre_transform, dataset, model, params) reduction(+: stats)
    for(size_t i=0; i<dataset.points.size(); i++)
    {
        // figure out if distance is too high
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
                stats += CrossStatistics::Init(Di, Mi);
            }
        }
    }

    return stats;
}

} // namespace rmagine 