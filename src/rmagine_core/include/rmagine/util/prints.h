/*
 * Copyright (c) 2022, University Osnabrück
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
 * @brief Prints for Rmagine math types
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_UTIL_PRINTS_H
#define RMAGINE_UTIL_PRINTS_H

#include <iostream>
#include <rmagine/math/types.h>
#include <rmagine/types/Memory.hpp>
#include <typeinfo>       // operator typeid


template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::Vector2_<DataT>& v)
{
    os << "v[" << v.x << "," << v.y << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::Vector3_<DataT>& v)
{
    os << "v[" << v.x << "," << v.y << "," << v.z << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::AABB_<DataT>& aabb)
{
    os << "AABB [" << aabb.min <<  " - " << aabb.max << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::Quaternion_<DataT>& q)
{
    os << "q[" << q.x << "," << q.y << "," << q.z << "," << q.w << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::EulerAngles_<DataT>& e)
{
    os << "E[" << e.roll << ", " << e.pitch << ", " << e.yaw << "]";
    return os;
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
inline std::ostream& operator<<(std::ostream& os, const rmagine::Matrix_<DataT, Rows, Cols>& M)
{
    os << "M" << M.rows() << "x" << M.cols() << "[\n";
    for(unsigned int i = 0; i < M.rows(); i++)
    {
        for(unsigned int j = 0; j < M.cols(); j++)
        {
            os << " " << M(i, j);
        }
        os << "\n";
    }
    os << "]";
    return os;
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
inline std::ostream& operator<<(std::ostream& os, const rmagine::MatrixSlice_<DataT, Rows, Cols>& M)
{
    os << "M" << M.rows() << "x" << M.cols() << "[\n";
    for(unsigned int i = 0; i < M.rows(); i++)
    {
        for(unsigned int j = 0; j < M.cols(); j++)
        {
            os << " " << M(i, j);
        }
        os << "\n";
    }
    os << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::Transform_<DataT>& T)
{
    rmagine::EulerAngles_<DataT> e;
    e.set(T.R);
    os << "T[" << T.t << ", " << e << "]";
    return os;
}

template<typename DataT>
inline std::ostream& operator<<(
  std::ostream& os, const rmagine::Gaussian1D_<DataT>& g)
{
  os << "Gaussian1D: \n";;
  os << "- mean: " << g.mean << "\n";
  os << "- sigma: " << g.sigma << "\n";
  os << "- n meas: " << g.n_meas << "\n";
  return os;
}

template<typename DataT>
inline std::ostream& operator<<(
  std::ostream& os, const rmagine::Gaussian2D_<DataT>& g)
{
  os << "Gaussian2D: \n";;
  os << "- mean: " << g.mean << "\n";
  os << "- sigma: \n";
  os << g.sigma << "\n";
  os << "- n meas: " << g.n_meas << "\n";
  return os;
}

template<typename DataT>
inline std::ostream& operator<<(
  std::ostream& os, const rmagine::Gaussian3D_<DataT>& g)
{
  os << "Gaussian2D: \n";;
  os << "- mean: " << g.mean << "\n";
  os << "- sigma: \n";
  os << g.sigma << "\n";
  os << "- n meas: " << g.n_meas << "\n";
  return os;
}

template<typename DataT>
inline std::ostream& operator<<(
  std::ostream& os, const rmagine::CrossStatistics_<DataT>& stats)
{
  os << "CrossStatistics: \n";;
  os << "- dataset mean: " << stats.dataset_mean << "\n";
  os << "- model mean: " << stats.model_mean << "\n";
  os << "- covariance: \n";
  os << stats.covariance << "\n";
  os << "- n meas: " << stats.n_meas << "\n";
  return os;
}

template<typename DataT, typename MemT>
inline std::ostream& operator<<(std::ostream& os, const rmagine::MemoryView<DataT, MemT> mem)
{
    os << "Mem (DataT=" << typeid(DataT).name() << ", MemT=" << typeid(MemT).name() 
        << ", Size=" << mem.size() << "): ";
    
    os << "[";
    if(mem.size() > 0)
    {
        rmagine::Memory<DataT, rmagine::RAM> first = mem(0, 1);
        os << first[0];

        if(mem.size() > 2)
        {
            os << ", ..., ";
        }

        if(mem.size() > 1)
        {
            rmagine::Memory<DataT, rmagine::RAM> last = mem(mem.size()-1, mem.size());
            os << last[0];
        }
    }
    os << "]";

    return os;
}


#endif // RMAGINE_UTIL_PRINTS_H