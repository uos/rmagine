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
 * @brief MatrixOps
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_MATRIX_OPS_HPP
#define RMAGINE_MATH_MATRIX_OPS_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

// Marker Types
struct MatrixMultInvalid {};
struct MatrixAddInvalid {};

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
class MatrixOps_
{
public:
  using MatrixAccess = MatrixAccess_<DataT, Rows, Cols>;
  using ThisType = MatrixOps_<DataT, Rows, Cols, MatrixAccess_>;

  using DataTConst = std::add_const_t<DataT>;
  using DataTNonConst = std::remove_const_t<DataT>;

  // MatrixOps_() = delete;

  ////////////////////
  // access functions
  RMAGINE_INLINE_FUNCTION
  DataT& at(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& at(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT at(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT at(unsigned int i, unsigned int j) volatile const;

  RMAGINE_INLINE_FUNCTION
  DataT& operator()(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& operator()(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT operator()(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT operator()(unsigned int row, unsigned int col) volatile const;

  //////////////////////////
  // initializer functions
  RMAGINE_INLINE_FUNCTION
  void setZeros();

  RMAGINE_INLINE_FUNCTION 
  void setOnes();

  RMAGINE_INLINE_FUNCTION 
  void setIdentity();

  /////////////////////
  // math functions
  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> negate() const;

  RMAGINE_INLINE_FUNCTION
  void negateInplace();

  template<unsigned int Cols2>
  RMAGINE_INLINE_FUNCTION 
  Matrix_<DataTNonConst, Rows, Cols2> mult(
    const Matrix_<DataT, Cols, Cols2>& M) const;

  RMAGINE_INLINE_FUNCTION 
  void multInplace(const Matrix_<DataT, Rows, Cols>& M);

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> mult(const DataT& scalar) const;

  RMAGINE_INLINE_FUNCTION
  void multInplace(const DataT& scalar);

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> multEwise(const Matrix_<DataT, Rows, Cols>& M) const;

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> div(const DataT& scalar) const;

  RMAGINE_INLINE_FUNCTION
  void divInplace(const DataT& scalar);

  RMAGINE_INLINE_FUNCTION 
  Vector3_<DataTNonConst> mult(const Vector3_<DataT>& v) const;

  RMAGINE_INLINE_FUNCTION 
  Vector2_<DataTNonConst> mult(const Vector2_<DataT>& v) const;

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> add(const Matrix_<DataT, Rows, Cols>& M) const;

  RMAGINE_INLINE_FUNCTION
  void addInplace(const Matrix_<DataT, Rows, Cols>& M);

  RMAGINE_INLINE_FUNCTION
  void addInplace(volatile Matrix_<DataT, Rows, Cols>& M) volatile;

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> sub(const Matrix_<DataT, Rows, Cols>& M) const;

  RMAGINE_INLINE_FUNCTION
  void subInplace(const Matrix_<DataT, Rows, Cols>& M);

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Cols, Rows> transpose() const;

  RMAGINE_INLINE_FUNCTION
  void transposeInplace();

  RMAGINE_INLINE_FUNCTION
  DataT trace() const;

  RMAGINE_INLINE_FUNCTION
  DataT det() const;

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Cols, Rows> inv() const;

  template<unsigned int RowsNew, unsigned int ColsNew>
  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, RowsNew, ColsNew> copy_block(unsigned int row, unsigned int col) const;

  template<unsigned int RowsBlock, unsigned int ColsBlock>
  RMAGINE_INLINE_FUNCTION
  void set_block(unsigned int row, unsigned int col, const Matrix_<DataT, RowsBlock, ColsBlock>& block);

  /////////////////////
  // math function aliases

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Cols, Rows> T() const 
  {
    return transpose();
  }

  ////////////////////
  // math function operators

  template<unsigned int Cols2>
  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols2> operator*(const Matrix_<DataT, Cols, Cols2>& M) const
  {
    return mult(M);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataT, Rows, Cols>& operator*=(const Matrix_<DataT, Rows, Cols>& M)
  {
    static_assert(Rows == Cols);
    multInplace(M);
    return static_cast<MatrixAccess&>(*this);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator*(const DataT& s) const
  {
    return mult(s);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataT, Rows, Cols>& operator*=(const DataT& s)
  {
    multInplace(s);
    return static_cast<MatrixAccess&>(*this);
  }

  RMAGINE_INLINE_FUNCTION
  Vector3_<DataTNonConst> operator*(const Vector3_<DataT>& p) const
  {
    return mult(p);
  }

  RMAGINE_INLINE_FUNCTION
  Vector2_<DataTNonConst> operator*(const Vector2_<DataT>& p) const
  {
    return mult(p);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator/(const DataT& s) const
  {
    return div(s);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataT, Rows, Cols>& operator/=(const DataT& s)
  {
    divInplace(s);
    return static_cast<MatrixAccess&>(*this);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator+(const Matrix_<DataT, Rows, Cols>& M) const
  {
    return add(M);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataT, Rows, Cols>& operator+=(const Matrix_<DataT, Rows, Cols>& M)
  {
    addInplace(M);
    return static_cast<MatrixAccess&>(*this);
  }

  RMAGINE_INLINE_FUNCTION
  volatile Matrix_<DataT, Rows, Cols>& operator+=(volatile Matrix_<DataT, Rows, Cols>& M) volatile
  {
    addInplace(M);
    return static_cast<volatile MatrixAccess&>(*this);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator-(const Matrix_<DataT, Rows, Cols>& M) const
  {
    return sub(M);
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator-() const
  {
    return negate();
  }

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> operator~() const
  {
    return inv();
  }

  /////////////////////
  // Transformation Helpers. Matrix Form: Square, Homogenous

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows-1, Cols-1> rotation() const;

  RMAGINE_INLINE_FUNCTION
  void setRotation(const Matrix_<DataT, Rows-1, Cols-1>& R);

  RMAGINE_INLINE_FUNCTION
  void setRotation(const Quaternion_<DataT>& q);

  RMAGINE_INLINE_FUNCTION
  void setRotation(const EulerAngles_<DataT>& e);

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows-1, 1> translation() const;

  RMAGINE_INLINE_FUNCTION
  void setTranslation(const Matrix_<DataT, Rows-1, 1>& t);

  RMAGINE_INLINE_FUNCTION
  void setTranslation(const Vector2_<DataT>& t);

  RMAGINE_INLINE_FUNCTION
  void setTranslation(const Vector3_<DataT>& t);

  RMAGINE_INLINE_FUNCTION
  Matrix_<DataTNonConst, Rows, Cols> invRigid() const;

  // const input
  template<template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_ >
  RMAGINE_INLINE_FUNCTION
  void set(const OtherMatrixAccess_<DataTConst, Rows, Cols>& other);

  // non const input
  template<template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_ >
  RMAGINE_INLINE_FUNCTION
  void set(const OtherMatrixAccess_<DataTNonConst, Rows, Cols>& other);

  RMAGINE_INLINE_FUNCTION
  void set(const Quaternion_<DataT>& q);

  RMAGINE_INLINE_FUNCTION
  void set(const EulerAngles_<DataT>& e);

  RMAGINE_INLINE_FUNCTION
  void set(const Transform_<DataT>& T);


  // const input
  template<template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_ >
  RMAGINE_INLINE_FUNCTION
  MatrixAccess_<DataT, Rows, Cols>& operator=(
    const OtherMatrixAccess_<DataTConst, Rows, Cols>& other)
  {
    // std::cout << "-- copy assign" << std::endl;
    set(other);
    return static_cast<MatrixAccess&>(*this);
  }

  // non const input
  template<template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_ >
  RMAGINE_INLINE_FUNCTION
  MatrixAccess_<DataT, Rows, Cols>& operator=(
    const OtherMatrixAccess_<DataTNonConst, Rows, Cols>& other)
  {
    // std::cout << "-- copy assign" << std::endl;
    set(other);
    return static_cast<MatrixAccess&>(*this);
  }

  template<template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_ >
  RMAGINE_INLINE_FUNCTION
  MatrixAccess_<DataT, Rows, Cols>& operator=(
    OtherMatrixAccess_<DataT, Rows, Cols>&& other)
  {
    // std::cout << "-- move assign" << std::endl;
    set(other);
    return static_cast<MatrixAccess&>(*this);
  }

  ////////////////
  // CASTS

  /**
   * @brief Matrix<2,1> -> Vector2
   * 
   * @return Vector2_<DataT> 
   */
  RMAGINE_INLINE_FUNCTION
  operator Vector2_<DataTNonConst>() const;

  /**
   * @brief Matrix<3,1> -> Vector3
   * 
   * @return Vector3_<DataT> 
   */
  RMAGINE_INLINE_FUNCTION
  operator Vector3_<DataTNonConst>() const;

  /**
   * @brief Rotation Matrix -> Quaternion
   * 
   * @return Quaternion_<DataT> 
   */
  RMAGINE_INLINE_FUNCTION
  operator Quaternion_<DataTNonConst>() const;

  /**
   * @brief Rotation Matrix -> EulerAngles
   * 
   * @return EulerAngles_<DataT> 
   */
  RMAGINE_INLINE_FUNCTION
  operator EulerAngles_<DataTNonConst>() const;

  /**
   * @brief Transformation Matrix -> Transform
   * WARNING: The matrix has to be isometric, i.e. composed only of
   * rotational and translational components. If it has e.g. scalar
   * components use the "decompose" function instead
   * 
   * @return Transform_<DataT>
   */
  RMAGINE_INLINE_FUNCTION
  operator Transform_<DataTNonConst>() const;

  /**
   * @brief Data Type Cast to ConvT
   * 
   * @tparam ConvT 
   * @return RMAGINE_INLINE_FUNCTION 
   */
  template<typename ConvT>
  RMAGINE_INLINE_FUNCTION
  Matrix_<ConvT, Rows, Cols> cast() const;

  /////////////////////
  // compile time functions

  // static template functions
  static constexpr unsigned int rows()
  {
      return Rows;
  }

  static constexpr unsigned int cols()
  {
      return Cols;
  }

  template<typename MatrixT>
  using MultResultType = typename std::conditional<
                          Cols == MatrixT::rows(),
                          Matrix_<DataTNonConst, Rows, MatrixT::cols()>, // Valid Multiplication
                          MatrixMultInvalid>::type;

  template<typename MatrixT>
  using AddResultType = typename std::conditional<
                          Cols == MatrixT::cols() && Rows == MatrixT::rows(),
                          Matrix_<DataTNonConst, Rows, MatrixT::cols()>, // Valid Multiplication
                          MatrixAddInvalid>::type;

  using Type = DataT;
};

} // namespace rmagine

#include "MatrixOps.tcc"

#endif // RMAGINE_MATH_MATRIX_OPS_HPP