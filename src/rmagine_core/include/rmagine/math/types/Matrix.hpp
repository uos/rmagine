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
 * @brief Matrix
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_MATRIX_HPP
#define RMAGINE_MATH_MATRIX_HPP


#include <rmagine/types/shared_functions.h>
#include <type_traits>

#include "definitions.h"

// TODO: remove
#include <iostream>

namespace rmagine
{


template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
class MatrixSlice_
: public MatrixOps_<DataT, Rows, Cols, Stride, MatrixSlice_>
{
public:
  using MatrixOps = MatrixOps_<DataT, Rows, Cols, Stride, MatrixSlice_>;
  using ThisType = MatrixSlice_<DataT, Rows, Cols, Stride>;

  // default constructor
  MatrixSlice_() = delete;

  // non-default constructor
  MatrixSlice_(DataT* data, const unsigned int row, const unsigned int col);

  using MatrixOps::operator=;

  MatrixSlice_<DataT, Rows, Cols, Stride>& operator=(
    const MatrixSlice_<DataT, Rows, Cols, Stride>& other)
  {
    return MatrixOps::operator=(other);
  }

  MatrixSlice_<DataT, Rows, Cols, Stride>& operator=(
    const MatrixSlice_<DataT, Rows, Cols, Stride>&& other)
  {
    return MatrixOps::operator=(other);
  }

  ////////////////////
  // access functions
  RMAGINE_INLINE_FUNCTION
  DataT& access(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& access(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT access(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT access(unsigned int i, unsigned int j) volatile const;

  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  MatrixSlice_<DataT, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col);

  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  const MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col) const;

protected:

  // DATA
  DataT* data;
  unsigned int row_offset;
  unsigned int col_offset;
};


template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
class Matrix_
: public MatrixOps_<DataT, Rows, Cols, Stride, Matrix_>
{
public:

  using MatrixOps = MatrixOps_<DataT, Rows, Cols, Stride, Matrix_>;
  using ThisType = Matrix_<DataT, Rows, Cols, Stride>;

  ////////////////////
  // access functions
  RMAGINE_INLINE_FUNCTION
  DataT& access(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& access(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT access(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT access(unsigned int i, unsigned int j) volatile const;

  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  MatrixSlice_<DataT, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col);

  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  const MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col) const;

  RMAGINE_FUNCTION
  static Matrix_<DataT, Rows, Cols, Stride> Zeros()
  {
    Matrix_<DataT, Rows, Cols, Stride> ret;
    ret.setZeros();
    return ret;
  }

  RMAGINE_FUNCTION
  static Matrix_<DataT, Rows, Cols, Stride> Ones()
  {
    Matrix_<DataT, Rows, Cols, Stride> ret;
    ret.setOnes();
    return ret;
  }

  RMAGINE_FUNCTION
  static Matrix_<DataT, Rows, Cols, Stride> Identity()
  {
    Matrix_<DataT, Rows, Cols, Stride> ret;
    ret.setIdentity();
    return ret;
  }

protected:
  // DATA
  DataT data[Cols * Stride];
};

} // namespace rmagine

#include "Matrix.tcc"

#include "MatrixOps.hpp"



#include "EulerAngles.hpp"
// TODO: put the following functions somewhere else

namespace rmagine
{

RMAGINE_INLINE_FUNCTION
Matrix_<float, 2, 2> yaw_to_rot_mat_2d(float yaw)
{
    Matrix_<float, 2, 2> R;
    R(0,0) =  cos(yaw); R(0,1) = sin(yaw);
    R(1,0) = -sin(yaw); R(1,1) = cos(yaw);
    return R;
}

RMAGINE_INLINE_FUNCTION
Matrix_<float, 3, 3> yaw_to_rot_mat_3d(float yaw)
{
    EulerAngles_<float> e = {0.0, 0.0, yaw};
    return static_cast<Matrix_<float, 3, 3> >(e);
}

} // namespace rmagine

#endif // RMAGINE_MATH_MATRIX_HPP