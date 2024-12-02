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
 * @brief Gaussian1D
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_GAUSSIAN_3D_HPP
#define RMAGINE_MATH_GAUSSIAN_3D_HPP

#include "definitions.h"

namespace rmagine
{

template<typename DataT>
struct Gaussian3D_
{
    Vector3_<DataT>       mean;
    Matrix_<DataT, 3, 3>  sigma;
    uint32_t n_meas;

    RMAGINE_FUNCTION
    static Gaussian3D_<DataT> Identity()
    {
        Gaussian3D_<DataT> ret;
        ret.mean = {0.0, 0.0, 0.0};
        ret.sigma.setZeros();
        ret.n_meas = 0; // never measured
        return ret;
    }

    RMAGINE_FUNCTION
    static Gaussian3D_<DataT> Init(
        const Vector3_<DataT>& measurement)
    {
        Gaussian3D_<DataT> ret;
        ret.mean = measurement;
        ret.sigma.setZeros();
        ret.n_meas = 1;
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    Gaussian3D_<DataT> add(const Gaussian3D_<DataT>& o) const;

    RMAGINE_INLINE_FUNCTION
    Gaussian3D_<DataT> operator+(const Gaussian3D_<DataT>& o) const
    {
        return add(o);
    }

    RMAGINE_INLINE_FUNCTION
    Gaussian3D_<DataT> operator+=(const Gaussian3D_<DataT>& o)
    { 
        const Gaussian3D_<DataT> res = add(o);
        mean = res.mean;
        sigma = res.sigma;
        n_meas = res.n_meas;
        return *this;
    }
};

} // namespace rmagine

#include "Gaussian3D.tcc"

#endif // RMAGINE_MATH_GAUSSIAN_3D_HPP