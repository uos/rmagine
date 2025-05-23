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
 * @brief Statistics Functions - CUDA
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_STATISTICS_CUH
#define RMAGINE_MATH_STATISTICS_CUH


#include <rmagine/math/statistics.h>
#include <rmagine/types/MemoryCuda.hpp>

// CUDA extension for statistics.h

namespace rmagine
{

/**
 * @brief Reducing dataset and model to the cross statistics using point to point (P2P) distances. CUDA version.
 * 
 * The N dataset and model points are reduced to a single cross statistic. 
 * The resulting statistical parameters can passed to a SVD to estimate the transformation between dataset and model (Umeyama).
 * There several options to mask dataset and model
 * - set mask of dataset or model to 0 at index=X to ignore the correspondence X
 * - set set id view of dataset or model to enable filtering by ID, which can be set via 'UmeyamaReductionConstraints'.
 * - set max_dist of UmeyamaReductionConstraints for a maximum allowed distance
 * 
 * It differs slightly from the CPU version as you can provide statistics view with a size > 1.
 * In that case the computation is splitted to N=size blocks on the GPU. The statistics view is then filled with the 
 * intermediate results wich can be later reduced to a total result.
 * 
 * @param[in] pre_transform Use this transformation to pretransform the dataset. set to Transform::Identity() for no transform.
 * @param[in] dataset       PointCloudView pointing to buffers of the dataset
 * @param[in] model         PointCloudView pointing to buffers of the model
 * @param[in] params        Constraints that need to be satisfied to 
 * @param[out] stats        Resulting statistics
 * 
 * @warning This GPU version has internally a different structure to compute the reduction.
 *          It shows a better numerical accuracy then the CPU conterpart.
 */
void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats);

void statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& statistics);

CrossStatistics statistics_p2p(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params);

/**
 * @brief Reducing dataset and model to the cross statistics using point to point (P2L) distances. CUDA version.
 * 
 * The N dataset and model points are reduced to a single cross statistic. 
 * The resulting statistical parameters can passed to a SVD to estimate the transformation between dataset and model (Umeyama).
 * There several options to mask dataset and model
 * - set mask of dataset or model to 0 at index=X to ignore the correspondence X
 * - set set id view of dataset or model to enable filtering by ID, which can be set via 'UmeyamaReductionConstraints'.
 * - set max_dist of UmeyamaReductionConstraints for a maximum allowed distance
 * 
 * It differs slightly from the CPU version as you can provide statistics view with a size > 1.
 * In that case the computation is splitted to N=size blocks on the GPU. The statistics view is then filled with the 
 * intermediate results wich can be later reduced to a total result.
 * 
 * @param[in] pre_transform Use this transformation to pretransform the dataset. set to Transform::Identity() for no transform.
 * @param[in] dataset       PointCloudView pointing to buffers of the dataset
 * @param[in] model         PointCloudView pointing to buffers of the model
 * @param[in] params        Constraints that need to be satisfied to 
 * @param[out] stats        Resulting statistics
 * 
 * @warning This GPU version has internally a different structure to compute the reduction.
 *          It shows a better numerical accuracy then the CPU conterpart.
 */
void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats);

void statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params,
    CrossStatistics& statistics);

CrossStatistics statistics_p2l(
    const Transform& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const UmeyamaReductionConstraints params);

void statistics_objectwise_p2l(
    const MemoryView<Transform, VRAM_CUDA>& pre_transform,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const unsigned int& width,
    const unsigned int& height,
    const MemoryView<UmeyamaReductionConstraints, VRAM_CUDA>& params,
    const MemoryView<AABB, VRAM_CUDA>& bboxes,
    MemoryView<CrossStatistics, VRAM_CUDA>& stats);


void statistics_objectwise_p2l(
    const MemoryView<Transform, RAM>& pre_transforms,
    const PointCloudView_<VRAM_CUDA>& dataset,
    const PointCloudView_<VRAM_CUDA>& model,
    const unsigned int& width,
    const unsigned int& height,
    const MemoryView<UmeyamaReductionConstraints, RAM>& params,
    const MemoryView<AABB, RAM>& bboxes,
    MemoryView<CrossStatistics, RAM>& stats);




} // namespace rmagine

#endif // RMAGINE_MATH_STATISTICS_CUH
