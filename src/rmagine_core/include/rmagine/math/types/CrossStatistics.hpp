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
 * @brief CrossStatistics
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_CROSS_STATISTICS_HPP
#define RMAGINE_MATH_CROSS_STATISTICS_HPP

#include "Vector3.hpp"
#include "Matrix.hpp"
#include <rmagine/types/shared_functions.h>
#include "definitions.h"

namespace rmagine
{

template<typename DataT>
struct CrossStatistics_
{
    Vector3_<DataT>       dataset_mean;
    Vector3_<DataT>       model_mean;
    Matrix_<DataT, 3, 3>  covariance;
    unsigned int          n_meas; // number of samples

    RMAGINE_FUNCTION
    static CrossStatistics_<DataT> Identity()
    {
        CrossStatistics_<DataT> ret;
        ret.dataset_mean = {0.0, 0.0, 0.0};
        ret.model_mean = {0.0, 0.0, 0.0};
        ret.covariance.setZeros();
        ret.n_meas = 0;
        return ret;
    }

    RMAGINE_FUNCTION
    static CrossStatistics_<DataT> Init(
        const Vector3_<DataT>& d, 
        const Vector3_<DataT>& m)
    {
        CrossStatistics_<DataT> ret;
        ret.dataset_mean = d;
        ret.model_mean = m;
        ret.covariance.setIdentity();
        ret.n_meas = 1;
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    CrossStatistics_<DataT> add(const CrossStatistics_<DataT>& o) const;

    RMAGINE_INLINE_FUNCTION
    CrossStatistics_<DataT> operator+(const CrossStatistics_<DataT>& o) const
    {
        return add(o);
    }

    RMAGINE_INLINE_FUNCTION
    void addInplace(const CrossStatistics_<DataT>& o);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile CrossStatistics_<DataT>& o) volatile;

    RMAGINE_INLINE_FUNCTION
    CrossStatistics_<DataT>& operator+=(const CrossStatistics_& o)
    {
        addInplace(o);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    volatile CrossStatistics_<DataT>& operator+=(volatile CrossStatistics_<DataT>& o) volatile
    {
        addInplace(o);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    void addInplace2(const CrossStatistics_<DataT>& o);
};

using CrossStatistics = CrossStatistics_<float>;

} // namespace rmagine

#include "CrossStatistics.tcc"

#endif // RMAGINE_MATH_CROSS_STATISTICS_HPP
