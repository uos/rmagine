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
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_STATISTICS_H
#define RMAGINE_MATH_STATISTICS_H

#include "types.h"
#include "math.h"
#include <rmagine/types/shared_functions.h>
#include "linalg.h"
#include <unordered_map>


namespace rmagine
{

struct UmeyamaReductionConstraints 
{
    ///
    // Ignore all correspondences larger than `max_dist`
    float max_dist;
    
    ///
    // Ignore dataset ids except for `dataset_id`
    unsigned int dataset_id;
    
    /// 
    // Ignore model ids except for `model_id`
    unsigned int model_id;
};

/**
 * @brief Reducing dataset and model to the cross statistics using point to point (P2P) distances.
 * 
 * The N dataset and model points are reduced to a single cross statistic. 
 * The resulting statistical parameters can passed to a SVD to estimate the transformation between dataset and model (Umeyama).
 * There several options to mask dataset and model
 * - set mask of dataset or model to 0 at index=X to ignore the correspondence X
 * - set set id view of dataset or model to enable filtering by ID, which can be set via 'UmeyamaReductionConstraints'.
 * - set max_dist of UmeyamaReductionConstraints for a maximum allowed distance
 * 
 * See: "MICP-L: Mesh-based ICP for Robot Localization using Hardware-Accelerated Ray Casting"
 * 
 * @param[in] pre_transform use this transformation to pretransform the dataset. set to Transform::Identity() for no transform.
 * @param[in] dataset  PointCloudView pointing to buffers of the dataset
 * @param[in] model    PointCloudView pointing to buffers of the model
 * @param[in] params   Constraints that need to be satisfied to 
 * @param[out] stats    Resulting statistics
 */
void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats);

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats);

CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params);

/**
 * @brief Reducing dataset and model to the cross statistics using point to plane (P2L) distances.
 * 
 * The N dataset and model points are reduced to a single cross statistic. 
 * The resulting statistical parameters can passed to a SVD to estimate the transformation between dataset and model (Umeyama).
 * There several options to mask dataset and model
 * - set mask of dataset or model to 0 at index=X to ignore the correspondence X
 * - set set id view of dataset or model to enable filtering by ID, which can be set via 'UmeyamaReductionConstraints'.
 * - set max_dist of UmeyamaReductionConstraints for a maximum allowed distance
 * 
 * See: "MICP-L: Mesh-based ICP for Robot Localization using Hardware-Accelerated Ray Casting"
 * 
 * @param[in] pre_transform use this transformation to pretransform the dataset. set to Transform::Identity() for no transform.
 * @param[in] dataset  PointCloudView pointing to buffers of the dataset
 * @param[in] model    PointCloudView pointing to buffers of the model
 * @param[in] params   Constraints that need to be satisfied to 
 * @param[out] stats    Resulting statistics
 */
void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats);

void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& stats);

CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const UmeyamaReductionConstraints params);


/**
 * @brief Find out cross statistics between dataset and model, where the model consists of several objects (intances or geometries)
 * 
 * Given an unsegmented scan and an initial guess of the scene, we can compute a cross statistic between the scan and the model's objects.
 * It returns one cross statistic per object id. Which can be used to deform the scene in a computationally efficient manner.
 * See: "Mesh-based Object Tracking for Dynamic Semantic 3D Scene Graphs via Ray Tracing"
 * 
 * This version requires a pre-allocated view of the correct size. For a more dynamic variant use the statistics_p2l_ow with the map return.
 * 
 * MemoryView<CrossStatistics>& stats:
 * it is most efficient for the object ids to be dense in order to not waste memory
 * e.g model_object_ids could look like this:
 * [-,-,-,0,2,-,-,-,1,-,-,-,-,3] -> requires 3 memory
 * but not like this
 * [-,-,-,1000,-,1] -> requires 1000 memory
 * 
 * @param pre_transform
 * @param dataset 
 * @param model 
 * @param params 
 * @param[out] stats A list of stats
 */
void statistics_p2l_ow(
    const PointCloudView_<RAM>& dataset,
    const PointCloudView_<RAM>& model,
    const MemoryView<Transform>& model_pretransforms,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics>& stats);

/**
 * @brief Find out cross statistics between dataset and model, where the model consists of several objects (intances or geometries)
 * 
 * Given an unsegmented scan and an initial guess of the scene, we can compute a cross statistic between the scan and the model's objects.
 * It returns one cross statistic per object id. Which can be used to deform the scene in a computationally efficient manner.
 * See: "Mesh-based Object Tracking for Dynamic Semantic 3D Scene Graphs via Ray Tracing"
 * 
 * @param pre_transform
 * @param dataset 
 * @param model 
 * @param params 
 * @return std::unordered_map<unsigned int, CrossStatistics> 
 */
// std::unordered_map<unsigned int, CrossStatistics> statistics_p2l_ow(
//     const PointCloudView_<RAM>& dataset,
//     const Transform& pre_transform,
//     const PointCloudView_<RAM>& model,
//     const UmeyamaReductionConstraints params);


} // namespace rmagine

#endif // RMAGINE_MATH_STATISTICS_H