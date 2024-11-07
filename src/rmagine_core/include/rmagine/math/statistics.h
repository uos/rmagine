/*
 * Copyright (c) 2024, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Statistics Functions
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_STATISTICS_H
#define RMAGINE_MATH_STATISTICS_H

#include "types.h"
#include "math.h"
#include <rmagine/types/shared_functions.h>
#include "linalg.h"


namespace rmagine
{

struct UmeyamaReductionParams 
{
    float max_dist;
};

RMAGINE_FUNCTION
template<typename DataT>
CrossStatistics_<DataT> merge(
    const CrossStatistics_<DataT>& stats_a, 
    const CrossStatistics_<DataT>& stats_b)
{
    CrossStatistics_<DataT> ret;
    ret.n_meas = stats_a.n_meas + stats_b.n_meas;
    
    const DataT w1 = static_cast<DataT>(stats_a.n_meas) / static_cast<DataT>(ret.n_meas);
    const DataT w2 = static_cast<DataT>(stats_b.n_meas) / static_cast<DataT>(ret.n_meas);

    ret.dataset_mean = stats_a.dataset_mean * w1 + stats_b.dataset_mean * w2;
    ret.model_mean = stats_a.model_mean * w1 + stats_b.model_mean * w2;

    const Matrix_<DataT, 3,3> P1 = stats_a.covariance * w1 + stats_b.covariance * w2;
    const Matrix_<DataT, 3,3> P2 = (stats_a.model_mean - ret.model_mean).multT(stats_a.dataset_mean - ret.dataset_mean) * w1 
                                 + (stats_b.model_mean - ret.model_mean).multT(stats_b.dataset_mean - ret.dataset_mean) * w2;
    ret.covariance = P1 + P2;
    return ret;
}

RMAGINE_FUNCTION
template<typename DataT>
CrossStatistics_<DataT> add_correspondence(
    const CrossStatistics_<DataT>& stats, 
    const Vector& d, 
    const Vector& m)
{
    CrossStatistics_<DataT> ret;
    ret.n_meas = stats.n_meas + 1;
    
    const DataT w1 = static_cast<float>(stats.n_meas) / static_cast<float>(ret.n_meas);
    const DataT w2 = 1.0 / static_cast<float>(ret.n_meas);

    ret.dataset_mean = stats.dataset_mean * w1 + d * w2;
    ret.model_mean = stats.model_mean * w1 + m * w2;

    const Matrix_<DataT, 3, 3> P1 = (m - ret.model_mean).multT(d - ret.dataset_mean);
    const Matrix_<DataT, 3, 3> P2 = (stats.model_mean - ret.model_mean).multT(stats.dataset_mean - ret.dataset_mean);

    ret.covariance = stats.covariance * w1 + P1 * w2 + P2 * w1;
    return ret;
}

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionParams params,
    CrossStatistics& statistics);

CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionParams params);

void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionParams params,
    CrossStatistics& statistics);

CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionParams params);

} // namespace rmagine

#endif // RMAGINE_MATH_STATISTICS_H