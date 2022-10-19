#ifndef RMAGINE_MATH_MATRIX_HPP
#define RMAGINE_MATH_MATRIX_HPP


#include <rmagine/types/shared_functions.h>
#include <iostream>
#include <type_traits>

namespace rmagine
{

// Move this to another header
template<typename DataT, unsigned int Rows, unsigned int Cols>
struct Matrix;


// Marker Types
struct MatrixMultInvalid {};
struct MatrixAddInvalid {};


template<typename DataT, unsigned int Rows, unsigned int Cols>
struct Matrix {
    DataT data[Cols][Rows];

    //////////////////////////
    // initializer functions

    RMAGINE_INLINE_FUNCTION 
    void setZeros();

    RMAGINE_INLINE_FUNCTION 
    void setOnes();

    RMAGINE_INLINE_FUNCTION 
    void setIdentity();

    ////////////////////
    // access functions

    RMAGINE_INLINE_FUNCTION
    DataT& at(unsigned int row, unsigned int col);

    RMAGINE_INLINE_FUNCTION
    DataT at(unsigned int row, unsigned int col) const;

    RMAGINE_INLINE_FUNCTION
    DataT* operator[](const unsigned int col);

    RMAGINE_INLINE_FUNCTION
    const DataT* operator[](const unsigned int col) const;

    RMAGINE_INLINE_FUNCTION
    DataT& operator()(unsigned int row, unsigned int col);

    RMAGINE_INLINE_FUNCTION
    DataT operator()(unsigned int row, unsigned int col) const;


    /////////////////////
    // math functions
    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> negate() const;

    RMAGINE_INLINE_FUNCTION
    void negateInplace();

    template<unsigned int Cols2>
    RMAGINE_INLINE_FUNCTION 
    Matrix<DataT, Rows, Cols2> mult(const Matrix<DataT, Cols, Cols2>& M) const;

    RMAGINE_INLINE_FUNCTION 
    void multInplace(const Matrix<DataT, Rows, Cols>& M);

    Matrix<DataT, Rows, Cols> mult(const DataT& scalar) const;

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> add(const Matrix<DataT, Rows, Cols>& M) const;

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> sub(const Matrix<DataT, Rows, Cols>& M) const;

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Cols, Rows> transpose() const;

    RMAGINE_INLINE_FUNCTION
    void transposeInplace();

    RMAGINE_INLINE_FUNCTION
    DataT trace() const;

    RMAGINE_INLINE_FUNCTION
    DataT det() const;

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Cols, Rows> inv() const;

    /////////////////////
    // math function aliases

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Cols, Rows> T() const 
    {
        return transpose();
    }

    ////////////////////
    // math function operators

    template<unsigned int Cols2>
    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols2> operator*(const Matrix<DataT, Cols, Cols2>& M) const
    {
        return mult(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> operator+(const Matrix<DataT, Rows, Cols>& M) const
    {
        return add(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> operator-(const Matrix<DataT, Rows, Cols>& M) const
    {
        return sub(M);
    }

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> operator-() const
    {
        return negate();
    }
    

    /////////////////////
    // Transformation Helpers. Matrix Form: Square, Homogenous

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows-1, Cols-1> rotation() const;

    RMAGINE_INLINE_FUNCTION
    void setRotation(const Matrix<DataT, Rows-1, Cols-1>& R);

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows-1, 1> translation() const;

    RMAGINE_INLINE_FUNCTION
    void setTranslation(const Matrix<DataT, Rows-1, 1>& t);

    RMAGINE_INLINE_FUNCTION
    Matrix<DataT, Rows, Cols> invRigid() const;

    // RMAGINE_INL


    /////////////////////
    // compile time functions

    RMAGINE_INLINE_FUNCTION
    void test();

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
                            Matrix<DataT, Rows, MatrixT::cols()>, // Valid Multiplication
                            MatrixMultInvalid>::type;

    template<typename MatrixT>
    using AddResultType = typename std::conditional<
                            Cols == MatrixT::cols() && Rows == MatrixT::rows(),
                            Matrix<DataT, Rows, MatrixT::cols()>, // Valid Multiplication
                            MatrixAddInvalid>::type;

    using Type = DataT;
};

} // namespace rmagine

#include "Matrix.tcc"

#endif // RMAGINE_MATH_MATRIX_HPP