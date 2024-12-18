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


namespace rmagine
{

// Marker Types
struct MatrixMultInvalid {};
struct MatrixAddInvalid {};

template<typename DataT, unsigned int Rows, unsigned int Cols>
struct MatrixData;

template<typename DataT, unsigned int Rows, unsigned int Cols>
class MatrixData 
{
public:
  using Type = MatrixData<DataT, Rows, Cols>;
  using ConstType = MatrixData<std::add_const_t<DataT>, Rows, Cols>;
  using VolatileType = MatrixData<std::add_volatile_t<DataT>, Rows, Cols>;

  using MatrixData_Type = Type;
  using MatrixData_ConstType = ConstType;
  using MatrixData_VolatileType = VolatileType;

  MatrixData() = default;

  ////////////////////
  // access functions
  RMAGINE_INLINE_FUNCTION
  DataT& at(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& at(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT at(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT at(unsigned int row, unsigned int col) volatile const;

  RMAGINE_INLINE_FUNCTION
  DataT& operator()(unsigned int row, unsigned int col);

  RMAGINE_INLINE_FUNCTION
  volatile DataT& operator()(unsigned int row, unsigned int col) volatile;

  RMAGINE_INLINE_FUNCTION
  DataT operator()(unsigned int row, unsigned int col) const;

  RMAGINE_INLINE_FUNCTION
  DataT operator()(unsigned int row, unsigned int col) volatile const;


  template<unsigned int SliceRows, unsigned int SliceCols>
  class Slice;

  // TODO: do we need a const pointer container?
  // template<unsigned int SliceRows, unsigned int SliceCols>
  // using ConstSlice = typename ConstType::Slice;


  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  Slice<SliceRows, SliceCols> slice(unsigned int row, unsigned int col);

  template<unsigned int SliceRows, unsigned int SliceCols>
  RMAGINE_INLINE_FUNCTION
  const Slice<SliceRows, SliceCols> slice(unsigned int row, unsigned int col) const
  {
    return Slice<SliceRows, SliceCols>(&data[0], row, col);
  }

  // template<unsigned int SliceRows, unsigned int SliceCols>
  // RMAGINE_INLINE_FUNCTION
  // ConstSlice<SliceRows, SliceCols> slice(unsigned int row, unsigned int col) const
  // {
  //   return ConstSlice<SliceRows, SliceCols>(&data[0], row, col);
  // }

  template<unsigned int SliceRows, unsigned int SliceCols>
  class Slice
  {
    public:
      Slice() = delete;
      explicit Slice(MatrixData<DataT, Rows, Cols>* data, 
        const unsigned int row,
        const unsigned int col);

      RMAGINE_INLINE_FUNCTION
      DataT& at(unsigned int row, unsigned int col);

      RMAGINE_INLINE_FUNCTION
      volatile DataT& at(unsigned int row, unsigned int col) volatile;

      RMAGINE_INLINE_FUNCTION
      DataT at(unsigned int row, unsigned int col) const;

      RMAGINE_INLINE_FUNCTION
      DataT at(unsigned int row, unsigned int col) volatile const;

      RMAGINE_INLINE_FUNCTION
      DataT& operator()(unsigned int row, unsigned int col);

      RMAGINE_INLINE_FUNCTION
      volatile DataT& operator()(unsigned int row, unsigned int col) volatile;

      RMAGINE_INLINE_FUNCTION
      DataT operator()(unsigned int row, unsigned int col) const;

      RMAGINE_INLINE_FUNCTION
      DataT operator()(unsigned int row, unsigned int col) volatile const;


      template<unsigned int SliceRowsNew, unsigned int SliceColsNew>
      RMAGINE_INLINE_FUNCTION
      Slice<SliceRowsNew, SliceColsNew> slice(unsigned int row, unsigned int col)
      {
        return Slice<SliceRowsNew, SliceColsNew>(data, row_offset + row, col_offset + col);
      }

      template<unsigned int SliceRowsNew, unsigned int SliceColsNew>
      RMAGINE_INLINE_FUNCTION
      const Slice<SliceRowsNew, SliceColsNew> slice(unsigned int row, unsigned int col) const
      {
        return Slice<SliceRowsNew, SliceColsNew>(data, row_offset + row, col_offset + col);
      }

    private:
      MatrixData<DataT, Rows, Cols>* data;
      const unsigned int row_offset;
      const unsigned int col_offset;
  };

private:
  DataT data[Cols*Rows];
};

template<typename DataT, unsigned int Rows, unsigned int Cols>
class Matrix_
{
public:
    // DATA
    DataT data[Cols*Rows];

    // using MatrixDataType = MatrixData<DataT, Rows, Cols>;

    //////////////////////////
    // initializer functions
    RMAGINE_INLINE_FUNCTION
    void setZeros();

    RMAGINE_INLINE_FUNCTION 
    void setOnes();

    RMAGINE_INLINE_FUNCTION 
    void setIdentity();

    RMAGINE_FUNCTION
    static Matrix_<DataT, Rows, Cols> Zeros()
    {
        Matrix_<DataT, Rows, Cols> ret;
        ret.setZeros();
        return ret;
    }

    RMAGINE_FUNCTION
    static Matrix_<DataT, Rows, Cols> Ones()
    {
        Matrix_<DataT, Rows, Cols> ret;
        ret.setOnes();
        return ret;
    }

    RMAGINE_FUNCTION
    static Matrix_<DataT, Rows, Cols> Identity()
    {
        Matrix_<DataT, Rows, Cols> ret;
        ret.setIdentity();
        return ret;
    }

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

    RMAGINE_INLINE_FUNCTION
    DataT* operator[](const unsigned int col);

    RMAGINE_INLINE_FUNCTION
    const DataT* operator[](const unsigned int col) const;

    /////////////////////
    // math functions
    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> negate() const;

    RMAGINE_INLINE_FUNCTION
    void negateInplace();

    template<unsigned int Cols2>
    RMAGINE_INLINE_FUNCTION 
    Matrix_<DataT, Rows, Cols2> mult(const Matrix_<DataT, Cols, Cols2>& M) const;

    RMAGINE_INLINE_FUNCTION 
    void multInplace(const Matrix_<DataT, Rows, Cols>& M);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> mult(const DataT& scalar) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const DataT& scalar);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> multEwise(const Matrix_<DataT, Rows, Cols>& M) const;

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> div(const DataT& scalar) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const DataT& scalar);

    RMAGINE_INLINE_FUNCTION 
    Vector3_<DataT> mult(const Vector3_<DataT>& v) const;

    RMAGINE_INLINE_FUNCTION 
    Vector2_<DataT> mult(const Vector2_<DataT>& v) const;

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> add(const Matrix_<DataT, Rows, Cols>& M) const;

    RMAGINE_INLINE_FUNCTION
    void addInplace(const Matrix_<DataT, Rows, Cols>& M);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile Matrix_<DataT, Rows, Cols>& M) volatile;

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> sub(const Matrix_<DataT, Rows, Cols>& M) const;

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Matrix_<DataT, Rows, Cols>& M);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Cols, Rows> transpose() const;

    RMAGINE_INLINE_FUNCTION
    void transposeInplace();

    RMAGINE_INLINE_FUNCTION
    DataT trace() const;

    RMAGINE_INLINE_FUNCTION
    DataT det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Cols, Rows> inv() const;

    template<unsigned int RowsNew, unsigned int ColsNew, unsigned int row, unsigned int col>
    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, RowsNew, ColsNew> copy_block() const;

    template<unsigned int RowsNew, unsigned int ColsNew>
    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, RowsNew, ColsNew> copy_block(unsigned int row, unsigned int col) const;

    template<unsigned int RowsBlock, unsigned int ColsBlock>
    RMAGINE_INLINE_FUNCTION
    void set_block(unsigned int row, unsigned int col, const Matrix_<DataT, RowsBlock, ColsBlock>& block);

    /////////////////////
    // math function aliases

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Cols, Rows> T() const 
    {
        return transpose();
    }

    ////////////////////
    // math function operators

    template<unsigned int Cols2>
    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols2> operator*(const Matrix_<DataT, Cols, Cols2>& M) const
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols>& operator*=(const Matrix_<DataT, Rows, Cols>& M)
    {
        static_assert(Rows == Cols);
        multInplace(M);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator*(const DataT& s) const
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols>& operator*=(const DataT& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const Vector3_<DataT>& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator*(const Vector2_<DataT>& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator/(const DataT& s) const
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols>& operator/=(const DataT& s)
    {
        divInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator+(const Matrix_<DataT, Rows, Cols>& M) const
    {
        return add(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols>& operator+=(const Matrix_<DataT, Rows, Cols>& M)
    {
        addInplace(M);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    volatile Matrix_<DataT, Rows, Cols>& operator+=(volatile Matrix_<DataT, Rows, Cols>& M) volatile
    {
        addInplace(M);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator-(const Matrix_<DataT, Rows, Cols>& M) const
    {
        return sub(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator-() const
    {
        return negate();
    }

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> operator~() const
    {
        return inv();
    }

    /////////////////////
    // Transformation Helpers. Matrix Form: Square, Homogenous

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows-1, Cols-1> rotation() const;

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Matrix_<DataT, Rows-1, Cols-1>& R);

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Quaternion_<DataT>& q);

    RMAGINE_INLINE_FUNCTION
    void setRotation(const EulerAngles_<DataT>& e);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows-1, 1> translation() const;

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Matrix_<DataT, Rows-1, 1>& t);

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Vector2_<DataT>& t);

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Vector3_<DataT>& t);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, Rows, Cols> invRigid() const;

    RMAGINE_INLINE_FUNCTION
    void set(const Quaternion_<DataT>& q);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles_<DataT>& e);

    RMAGINE_INLINE_FUNCTION
    void set(const Transform_<DataT>& T);

    ////////////////
    // CASTS

    /**
     * @brief Matrix<2,1> -> Vector2
     * 
     * @return Vector2_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Vector2_<DataT>() const;

    /**
     * @brief Matrix<3,1> -> Vector3
     * 
     * @return Vector3_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Vector3_<DataT>() const;

    /**
     * @brief Rotation Matrix -> Quaternion
     * 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Quaternion_<DataT>() const;

    /**
     * @brief Rotation Matrix -> EulerAngles
     * 
     * @return EulerAngles_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator EulerAngles_<DataT>() const;

    /**
     * @brief Transformation Matrix -> Transform
     * WARNING: The matrix has to be isometric, i.e. composed only of
     * rotational and translational components. If it has e.g. scalar
     * components use the "decompose" function instead
     * 
     * @return Transform_<DataT>
     */
    RMAGINE_INLINE_FUNCTION
    operator Transform_<DataT>() const;

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
                            Matrix_<DataT, Rows, MatrixT::cols()>, // Valid Multiplication
                            MatrixMultInvalid>::type;

    template<typename MatrixT>
    using AddResultType = typename std::conditional<
                            Cols == MatrixT::cols() && Rows == MatrixT::rows(),
                            Matrix_<DataT, Rows, MatrixT::cols()>, // Valid Multiplication
                            MatrixAddInvalid>::type;

    using Type = DataT;
};


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



#include "Matrix.tcc"

#endif // RMAGINE_MATH_MATRIX_HPP