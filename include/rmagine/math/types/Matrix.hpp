#ifndef RMAGINE_MATH_MATRIX_HPP
#define RMAGINE_MATH_MATRIX_HPP


#include <rmagine/types/shared_functions.h>
#include <type_traits>

#include "definitions.h"


namespace rmagine
{

// Move this to another header
template<typename DataT, unsigned int Rows, unsigned int Cols>
struct Matrix;


// Marker Types
struct MatrixMultInvalid {};
struct MatrixAddInvalid {};


template<typename DataT, unsigned int Rows, unsigned int Cols>
struct Matrix_ {
    // raw data
    DataT data[Cols][Rows];

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

} // namespace rmagine

#include "Matrix.tcc"

#endif // RMAGINE_MATH_MATRIX_HPP