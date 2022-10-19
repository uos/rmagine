#include "Matrix.hpp"

namespace rmagine {


template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col)
{
    return data[col][row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile
{
    return data[col][row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) const
{
    return data[col][row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile const
{
    return data[col][row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& Matrix_<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col)
{
    return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& Matrix_<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) volatile
{
    return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) const
{
    return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) volatile const
{
    return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT* Matrix_<DataT, Rows, Cols>::operator[](const unsigned int col) 
{
    return data[col];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
const DataT* Matrix_<DataT, Rows, Cols>::operator[](const unsigned int col) const 
{
    return data[col];
}

////////////////////
// setZeros
////////////////
template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<DataT, Rows, Cols>::setZeros()
{
    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            at(i, j) = static_cast<DataT>(0);
        }
    }
}

// specializations
template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<float, 3, 3>::setZeros()
{
    at(0,0) = 0.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 0.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 0.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<double, 3, 3>::setZeros()
{
    at(0,0) = 0.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 0.0;
    at(1,2) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 0.0;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<float, 4, 4>::setZeros()
{
    at(0,0) = 0.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(0,3) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 0.0f;
    at(1,2) = 0.0f;
    at(1,3) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 0.0f;
    at(2,3) = 0.0f;
    at(3,0) = 0.0f;
    at(3,1) = 0.0f;
    at(3,2) = 0.0f;
    at(3,3) = 0.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<double, 4, 4>::setZeros()
{
    at(0,0) = 0.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 0.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 0.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 0.0;
}


////////////////////
// setOnes
////////////////
template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<DataT, Rows, Cols>::setOnes()
{
    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            at(i, j) = static_cast<DataT>(1);
        }
    }
}

// specializations
template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<float, 3, 3>::setOnes()
{
    at(0,0) = 1.0f;
    at(0,1) = 1.0f;
    at(0,2) = 1.0f;
    at(1,0) = 1.0f;
    at(1,1) = 1.0f;
    at(1,2) = 1.0f;
    at(2,0) = 1.0f;
    at(2,1) = 1.0f;
    at(2,2) = 1.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<double, 3, 3>::setOnes()
{
    at(0,0) = 1.0;
    at(0,1) = 1.0;
    at(0,2) = 1.0;
    at(1,0) = 1.0;
    at(1,1) = 1.0;
    at(1,2) = 1.0;
    at(2,0) = 1.0;
    at(2,1) = 1.0;
    at(2,2) = 1.0;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<float, 4, 4>::setOnes()
{
    at(0,0) = 1.0f;
    at(0,1) = 1.0f;
    at(0,2) = 1.0f;
    at(0,3) = 1.0f;
    at(1,0) = 1.0f;
    at(1,1) = 1.0f;
    at(1,2) = 1.0f;
    at(1,3) = 1.0f;
    at(2,0) = 1.0f;
    at(2,1) = 1.0f;
    at(2,2) = 1.0f;
    at(2,3) = 1.0f;
    at(3,0) = 1.0f;
    at(3,1) = 1.0f;
    at(3,2) = 1.0f;
    at(3,3) = 1.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<double, 4, 4>::setOnes()
{
    at(0,0) = 1.0;
    at(0,1) = 1.0;
    at(0,2) = 1.0;
    at(0,3) = 1.0;
    at(1,0) = 1.0;
    at(1,1) = 1.0;
    at(1,2) = 1.0;
    at(1,3) = 1.0;
    at(2,0) = 1.0;
    at(2,1) = 1.0;
    at(2,2) = 1.0;
    at(2,3) = 1.0;
    at(3,0) = 1.0;
    at(3,1) = 1.0;
    at(3,2) = 1.0;
    at(3,3) = 1.0;
}

////////////////////
// setIdentity
////////////////
template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setIdentity()
{
    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            if(i == j)
            {
                at(i, j) = 1;
            } else {
                at(i, j) = 0;
            }
        }
    }
}

// specializatons
template<> 
RMAGINE_INLINE_FUNCTION
void Matrix_<float, 3, 3>::setIdentity()
{
    at(0,0) = 1.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 1.0f;
    at(1,2) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 1.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION
void Matrix_<double, 3, 3>::setIdentity()
{
    at(0,0) = 1.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 1.0;
    at(1,2) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 1.0;
}

template<> 
RMAGINE_INLINE_FUNCTION
void Matrix_<float, 4, 4>::setIdentity()
{
    at(0,0) = 1.0f;
    at(0,1) = 0.0f;
    at(0,2) = 0.0f;
    at(0,3) = 0.0f;
    at(1,0) = 0.0f;
    at(1,1) = 1.0f;
    at(1,2) = 0.0f;
    at(1,3) = 0.0f;
    at(2,0) = 0.0f;
    at(2,1) = 0.0f;
    at(2,2) = 1.0f;
    at(2,3) = 0.0f;
    at(3,0) = 0.0f;
    at(3,1) = 0.0f;
    at(3,2) = 0.0f;
    at(3,3) = 1.0f;
}

template<> 
RMAGINE_INLINE_FUNCTION
void Matrix_<double, 4, 4>::setIdentity()
{
    at(0,0) = 1.0;
    at(0,1) = 0.0;
    at(0,2) = 0.0;
    at(0,3) = 0.0;
    at(1,0) = 0.0;
    at(1,1) = 1.0;
    at(1,2) = 0.0;
    at(1,3) = 0.0;
    at(2,0) = 0.0;
    at(2,1) = 0.0;
    at(2,2) = 1.0;
    at(2,3) = 0.0;
    at(3,0) = 0.0;
    at(3,1) = 0.0;
    at(3,2) = 0.0;
    at(3,3) = 1.0;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> 
    Matrix_<DataT, Rows, Cols>::negate() const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            res(i, j) = -at(i, j);
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::negateInplace()
{
    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            at(i, j) = -at(i, j);
        }
    }
}


template<typename DataT, unsigned int Rows, unsigned int Cols> 
template<unsigned int Cols2>
RMAGINE_INLINE_FUNCTION 
Matrix_<DataT, Rows, Cols2> 
    Matrix_<DataT, Rows, Cols>::mult(const Matrix_<DataT, Cols, Cols2>& M) const
{
    // constexpr unsigned int Rows2 = Cols;
    constexpr unsigned int Rows3 = Rows;
    constexpr unsigned int Cols3 = Cols2;

    Matrix_<DataT, Rows3, Cols3> res;
    
    // before
    res.setZeros();
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            for(unsigned int k = 0; k < Cols2; k++)
            {
                res(i, k) += at(i, j) * M(j, k);
            }
        }
    }

    // both of the outer loops could be run asynchonously
    // - Test: slower than version on top
    //      tested in serial and with openmp
    // 
    // #pragma omp parallel for
    // for(unsigned int i = 0; i < Rows; i++)
    // {
    //     for(unsigned int k = 0; k < Cols2; k++)
    //     {
    //         res(i, k) = 0.0;
    //         for(unsigned int j = 0; j < Cols; j++)
    //         {
    //             res(i, k) += at(i, j) * M(j, k);
    //         }
    //     }
    // }
    // the second version 

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION 
void Matrix_<DataT, Rows, Cols>::multInplace(const Matrix_<DataT, Rows, Cols>& M)
{
    static_assert(Rows == Cols);

    // tmp memory
    
    // TODO: test
    // - processing each column should be thread safe
    // #pragma omp parallel for
    for(unsigned int j = 0; j < Cols; j++)
    {
        // copy entire column
        const DataT* col = data[j];
        DataT tmp[Rows];
        std::copy(col, col + Rows, tmp);

        for(unsigned int i = 0; i < Rows; i++)
        {
            at(i,j) = 0.0;
            for(unsigned int k = 0; k < Cols; k++)
            {
                at(i,j) += tmp[k] * M(i,k);
            }
        }
    }

}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> 
    Matrix_<DataT, Rows, Cols>::mult(const DataT& scalar) const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(i, j) = at(i, j) * scalar;
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::multInplace(const DataT& scalar)
{
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            at(i, j) *= scalar;
        }
    }
}


template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> 
    Matrix_<DataT, Rows, Cols>::div(const DataT& scalar) const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(i, j) = at(i, j) / scalar;
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::divInplace(const DataT& scalar)
{
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            at(i, j) /= scalar;
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION 
Vector3_<DataT> 
    Matrix_<DataT, Rows, Cols>::mult(const Vector3_<DataT>& v) const
{
    if constexpr(Rows == 3 && Cols == 3)
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z, 
            at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z, 
            at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z
        };
    } else 
    if constexpr(Rows == 3 && Cols == 4
                || Rows == 4 && Cols == 4)
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z + at(0,3),
            at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z + at(1,3),
            at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z + at(2,3)
        };
    }

    return {NAN, NAN, NAN};
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION 
Vector2_<DataT> 
    Matrix_<DataT, Rows, Cols>::mult(const Vector2_<DataT>& v) const
{
    if constexpr(Rows == 2 && Cols == 2)
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z, 
            at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z, 
            at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z
        };
    } else 
    if constexpr(Rows == 2 && Cols == 3
                || Rows == 3 && Cols == 3)
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z + at(0,3),
            at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z + at(1,3),
            at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z + at(2,3)
        };
    } else {
        return {NAN, NAN, NAN};
    }
    
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> 
    Matrix_<DataT, Rows, Cols>::add(const Matrix_<DataT, Rows, Cols>& M) const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(i, j) = at(i, j) + M(i, j);
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::addInplace(const Matrix_<DataT, Rows, Cols>& M)
{
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            at(i, j) += M(i, j);
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::addInplace(volatile Matrix_<DataT, Rows, Cols>& M) volatile
{
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            at(i, j) += M(i, j);
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> 
    Matrix_<DataT, Rows, Cols>::sub(const Matrix_<DataT, Rows, Cols>& M) const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(i, j) = at(i, j) - M(i, j);
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::subInplace(const Matrix_<DataT, Rows, Cols>& M)
{
    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            at(i, j) -= M(i, j);
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Cols, Rows> 
    Matrix_<DataT, Rows, Cols>::transpose() const
{
    Matrix_<DataT, Cols, Rows> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(j, i) = at(i, j);
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::transposeInplace()
{
    static_assert(Rows == Cols);
    float swap_mem;

    for(unsigned int i = 0; i < Rows - 1; i++)
    {
        for(unsigned int j = i + 1; j < Cols; j++)
        {
            swap_mem = at(i, j);
            at(i, j) = at(j, i);
            at(j, i) = swap_mem;
        }
    }
}

// specializations
template<> 
RMAGINE_INLINE_FUNCTION
void Matrix_<float, 3, 3>::transposeInplace()
{
    // use only one float as additional memory
    float swap_mem;
    // can we do this without additional memory?

    swap_mem = at(0,1);
    at(0,1) = at(1,0);
    at(1,0) = swap_mem;

    swap_mem = at(0,2);
    at(0,2) = at(2,0);
    at(2,0) = swap_mem;

    swap_mem = at(1,2);
    at(1,2) = at(2,1);
    at(2,1) = swap_mem;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::trace() const
{
    static_assert(Rows == Cols);

    DataT res = static_cast<DataT>(0);
    for(size_t i=0; i<Rows; i++)
    {
        res += at(i, i);
    }

    return res;
}

template<>
RMAGINE_INLINE_FUNCTION
float Matrix_<float, 3, 3>::det() const
{
    return  at(0, 0) * (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) -
            at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
            at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
}

template<> 
RMAGINE_INLINE_FUNCTION
float Matrix_<float, 4, 4>::det() const
{
    // TODO: check
    const float A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const float A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const float A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const float A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const float A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const float A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const float A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const float A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const float A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const float A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const float A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const float A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const float A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const float A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const float A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const float A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const float A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const float A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    return  at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
            - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
            + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
            - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );
}


template<> 
RMAGINE_INLINE_FUNCTION
Matrix_<float, 3, 3> Matrix_<float, 3, 3>::inv() const
{
    Matrix_<float, 3, 3> ret;

    const float invdet = 1 / det();

    ret(0, 0) = (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) * invdet;
    ret(0, 1) = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * invdet;
    ret(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * invdet;
    ret(1, 0) = (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * invdet;
    ret(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * invdet;
    ret(1, 2) = (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * invdet;
    ret(2, 0) = (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * invdet;
    ret(2, 1) = (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * invdet;
    ret(2, 2) = (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * invdet;

    return ret;
}

template<> 
RMAGINE_INLINE_FUNCTION
Matrix_<float, 4, 4> Matrix_<float, 4, 4>::inv() const
{
    // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    // answer of willnode at Jun 8 '17 at 23:09

    const float A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const float A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const float A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const float A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const float A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const float A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const float A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const float A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const float A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const float A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const float A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const float A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const float A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const float A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const float A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const float A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const float A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const float A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    float det_  = at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
                - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
                + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
                - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 ) ;

    // inv det
    det_ = 1.0f / det_;

    Matrix_<float, 4, 4> ret;
    ret(0,0) = det_ *   ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 );
    ret(0,1) = det_ * - ( at(0,1) * A2323 - at(0,2) * A1323 + at(0,3) * A1223 );
    ret(0,2) = det_ *   ( at(0,1) * A2313 - at(0,2) * A1313 + at(0,3) * A1213 );
    ret(0,3) = det_ * - ( at(0,1) * A2312 - at(0,2) * A1312 + at(0,3) * A1212 );
    ret(1,0) = det_ * - ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 );
    ret(1,1) = det_ *   ( at(0,0) * A2323 - at(0,2) * A0323 + at(0,3) * A0223 );
    ret(1,2) = det_ * - ( at(0,0) * A2313 - at(0,2) * A0313 + at(0,3) * A0213 );
    ret(1,3) = det_ *   ( at(0,0) * A2312 - at(0,2) * A0312 + at(0,3) * A0212 );
    ret(2,0) = det_ *   ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 );
    ret(2,1) = det_ * - ( at(0,0) * A1323 - at(0,1) * A0323 + at(0,3) * A0123 );
    ret(2,2) = det_ *   ( at(0,0) * A1313 - at(0,1) * A0313 + at(0,3) * A0113 );
    ret(2,3) = det_ * - ( at(0,0) * A1312 - at(0,1) * A0312 + at(0,3) * A0112 );
    ret(3,0) = det_ * - ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );
    ret(3,1) = det_ *   ( at(0,0) * A1223 - at(0,1) * A0223 + at(0,2) * A0123 );
    ret(3,2) = det_ * - ( at(0,0) * A1213 - at(0,1) * A0213 + at(0,2) * A0113 );
    ret(3,3) = det_ *   ( at(0,0) * A1212 - at(0,1) * A0212 + at(0,2) * A0112 );

    return ret;
}


/////////////////////
// Transformation Helpers

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows-1, Cols-1> Matrix_<DataT, Rows, Cols>::rotation() const
{
    static_assert(Rows == Cols);
    Matrix_<DataT, Rows-1, Cols-1> res;

    for(unsigned int i=0; i < Rows - 1; i++)
    {
        for(unsigned int j=0; j < Cols - 1; j++)
        {
            res(i, j) = at(i, j);
        }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setRotation(const Matrix_<DataT, Rows-1, Cols-1>& R)
{
    for(unsigned int i=0; i < Rows - 1; i++)
    {
        for(unsigned int j=0; j < Cols - 1; j++)
        {
            at(i, j) = R(i, j);
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setRotation(const Quaternion_<DataT>& q)
{
    static_assert(Rows >= 3 && Cols >= 3);
    Matrix_<DataT, 3, 3> R;
    R = q;
    setRotation(R);
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setRotation(const EulerAngles_<DataT>& e)
{
    static_assert(Rows >= 3 && Cols >= 3);
    Matrix_<DataT, 3, 3> R;
    R = e;
    setRotation(R);
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows-1, 1> Matrix_<DataT, Rows, Cols>::translation() const
{
    static_assert(Rows == Cols);
    Matrix_<DataT, Rows-1, 1> res;

    for(unsigned int i=0; i < Rows - 1; i++)
    {
        res(i, 0) = at(i, 0);
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setTranslation(const Matrix_<DataT, Rows-1, 1>& t)
{
    for(unsigned int i=0; i < Rows - 1; i++)
    {
        at(i, Cols-1) = t(i, 0);
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setTranslation(const Vector2_<DataT>& t)
{
    static_assert(Rows >= 2 && Cols >= 3);
    at(0, Cols-1) = t.x;
    at(1, Cols-1) = t.y;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::setTranslation(const Vector3_<DataT>& t)
{
    static_assert(Rows >= 3 && Cols >= 4);
    at(0, Cols-1) = t.x;
    at(1, Cols-1) = t.y;
    at(2, Cols-1) = t.z;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols> Matrix_<DataT, Rows, Cols>::invRigid() const
{
    static_assert(Rows == Cols);
    Matrix_<DataT, Rows, Cols> ret;
    ret.setIdentity();

    // TODO

    Matrix_<DataT, Rows-1, Cols-1> R = rotation();
    Matrix_<DataT, Rows-1, 1>      t = translation();

    R.transposeInplace();
    ret.setRotation(R);
    ret.setTranslation(- (R * t) );

    return ret;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::set(const Quaternion_<DataT>& q)
{
    static_assert(Rows >= 3);
    static_assert(Cols >= 3);
    
    at(0,0) = 2.0f * (q.w * q.w + q.x * q.x) - 1.0f;
    at(0,1) = 2.0f * (q.x * q.y - q.w * q.z);
    at(0,2) = 2.0f * (q.x * q.z + q.w * q.y);
    at(1,0) = 2.0f * (q.x * q.y + q.w * q.z);
    at(1,1) = 2.0f * (q.w * q.w + q.y * q.y) - 1.0f;
    at(1,2) = 2.0f * (q.y * q.z - q.w * q.x);
    at(2,0) = 2.0f * (q.x * q.z - q.w * q.y);
    at(2,1) = 2.0f * (q.y * q.z + q.w * q.x);
    at(2,2) = 2.0f * (q.w * q.w + q.z * q.z) - 1.0f;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::set(const EulerAngles_<DataT>& e)
{
    static_assert(Rows >= 3);
    static_assert(Cols >= 3);
    // Wrong?
    // TODO check
    // 1. test: correct

    const float cA = cosf(e.roll);
    const float sA = sinf(e.roll);
    const float cB = cosf(e.pitch);
    const float sB = sinf(e.pitch);
    const float cC = cosf(e.yaw);
    const float sC = sinf(e.yaw);

    at(0,0) =  cB * cC;
    at(0,1) = -cB * sC;
    at(0,2) =  sB;
   
    at(1,0) =  sA * sB * cC + cA * sC;
    at(1,1) = -sA * sB * sC + cA * cC;
    at(1,2) = -sA * cB;
    
    at(2,0) = -cA * sB * cC + sA * sC;
    at(2,1) =  cA * sB * sC + sA * cC;
    at(2,2) =  cA * cB;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::set(const Transform_<DataT>& T)
{
    static_assert(Rows >= 3);
    static_assert(Cols >= 4);
    setIdentity();
    setRotation(T.R);
    setTranslation(T.t);
}


} // namespace rmagine