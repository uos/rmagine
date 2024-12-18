#include "Matrix.hpp"

namespace rmagine {


///////////////////////////
/// MatrixData

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& MatrixData<DataT, Rows, Cols>::at(unsigned int row, unsigned int col)
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixData<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) const
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile const
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& MatrixData<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col)
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixData<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) volatile
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) const
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::operator()(unsigned int row, unsigned int col) volatile const
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols> MatrixData<DataT, Rows, Cols>::slice(
  unsigned int row, unsigned int col)
{
  return Slice<SliceRows, SliceCols>(&data[0], row, col);
}



///////////////////////////
/// MatrixData::Slice

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::Slice(
  MatrixData<DataT, Rows, Cols>* data, 
  const unsigned int row, 
  const unsigned int col)
:data(data)
,row_offset(row)
,col_offset(col)
{

}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT& MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::at(
  unsigned int row, unsigned int col)
{
  return data[(col_offset + col) * Rows + (row_offset + row)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::at(
  unsigned int row, unsigned int col) volatile
{
  return data[(col_offset + col) * Rows + (row_offset + row)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::at(
  unsigned int row, unsigned int col) const
{
  return data[(col_offset + col) * Rows + (row_offset + row)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::at(
  unsigned int row, unsigned int col) volatile const
{
  return data[(col_offset + col) * Rows + (row_offset + row)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT& MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::operator()(unsigned int row, unsigned int col)
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::operator()(unsigned int row, unsigned int col) volatile
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::operator()(unsigned int row, unsigned int col) const
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
DataT MatrixData<DataT, Rows, Cols>::Slice<SliceRows, SliceCols>::operator()(unsigned int row, unsigned int col) volatile const
{
  return at(row, col);
}















template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col)
{
    return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile
{
    return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) const
{
    return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::at(unsigned int row, unsigned int col) volatile const
{
    return data[col * Rows + row];
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
    return &data[col * Rows];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
const DataT* Matrix_<DataT, Rows, Cols>::operator[](const unsigned int col) const 
{
    return &data[col * Rows];
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
                at(i, j) = static_cast<DataT>(1);
            } else {
                at(i, j) = static_cast<DataT>(0);
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
        const DataT* col = &data[j * Rows];
        DataT tmp[Rows];
        for(unsigned int i = 0; i < Rows; i++)
        {
            tmp[i] = col[i];
        }

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
Matrix_<DataT, Rows, Cols> Matrix_<DataT, Rows, Cols>::multEwise(const Matrix_<DataT, Rows, Cols>& M) const
{
    Matrix_<DataT, Rows, Cols> res;

    for(unsigned int i = 0; i < Rows; i++)
    {
        for(unsigned int j = 0; j < Cols; j++)
        {
            res(i, j) = at(i, j) * M(i,j);
        }
    }

    return res;
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
            at(0,0) * v.x + at(0,1) * v.y, 
            at(1,0) * v.x + at(1,1) * v.y,
        };
    } else 
    if constexpr(Rows == 2 && Cols == 3
                || Rows == 3 && Cols == 3)
    {
        return {
            at(0,0) * v.x + at(0,1) * v.y + at(0,2),
            at(1,0) * v.x + at(1,1) * v.y + at(1,2),
        };
    } else {
        return {NAN, NAN};
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
float Matrix_<float, 2, 2>::det() const
{
    return at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);
}

template<>
RMAGINE_INLINE_FUNCTION
double Matrix_<double, 2, 2>::det() const
{
    return at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0);
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
double Matrix_<double, 3, 3>::det() const
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
double Matrix_<double, 4, 4>::det() const
{
    // TODO: check
    const double A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const double A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const double A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const double A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const double A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const double A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const double A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const double A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const double A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const double A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const double A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const double A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const double A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const double A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const double A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const double A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const double A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const double A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    return  at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
            - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
            + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
            - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 );
}

template<>
RMAGINE_INLINE_FUNCTION
Matrix_<float, 2, 2> Matrix_<float, 2, 2>::inv() const
{
    Matrix_<float, 2, 2> ret;
    
    const float invdet = 1.0f / det();
    ret(0, 0) =  at(1, 1) * invdet;
    ret(0, 1) = -at(0, 1) * invdet;
    ret(1, 0) = -at(1, 0) * invdet;
    ret(1, 1) =  at(0, 0) * invdet;
    
    return ret;
}

template<>
RMAGINE_INLINE_FUNCTION
Matrix_<double, 2, 2> Matrix_<double, 2, 2>::inv() const
{
    Matrix_<double, 2, 2> ret;
    
    const double invdet = 1.0 / det();
    ret(0, 0) =  at(1, 1) * invdet;
    ret(0, 1) = -at(0, 1) * invdet;
    ret(1, 0) = -at(1, 0) * invdet;
    ret(1, 1) =  at(0, 0) * invdet;
    
    return ret;
}

template<> 
RMAGINE_INLINE_FUNCTION
Matrix_<float, 3, 3> Matrix_<float, 3, 3>::inv() const
{
    Matrix_<float, 3, 3> ret;

    const float invdet = 1.0f / det();

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
Matrix_<double, 3, 3> Matrix_<double, 3, 3>::inv() const
{
    Matrix_<double, 3, 3> ret;

    const double invdet = 1.0 / det();

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


template<> 
RMAGINE_INLINE_FUNCTION
Matrix_<double, 4, 4> Matrix_<double, 4, 4>::inv() const
{
    // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
    // answer of willnode at Jun 8 '17 at 23:09

    const double A2323 = at(2,2) * at(3,3) - at(2,3) * at(3,2);
    const double A1323 = at(2,1) * at(3,3) - at(2,3) * at(3,1);
    const double A1223 = at(2,1) * at(3,2) - at(2,2) * at(3,1);
    const double A0323 = at(2,0) * at(3,3) - at(2,3) * at(3,0);
    const double A0223 = at(2,0) * at(3,2) - at(2,2) * at(3,0);
    const double A0123 = at(2,0) * at(3,1) - at(2,1) * at(3,0);
    const double A2313 = at(1,2) * at(3,3) - at(1,3) * at(3,2);
    const double A1313 = at(1,1) * at(3,3) - at(1,3) * at(3,1);
    const double A1213 = at(1,1) * at(3,2) - at(1,2) * at(3,1);
    const double A2312 = at(1,2) * at(2,3) - at(1,3) * at(2,2);
    const double A1312 = at(1,1) * at(2,3) - at(1,3) * at(2,1);
    const double A1212 = at(1,1) * at(2,2) - at(1,2) * at(2,1);
    const double A0313 = at(1,0) * at(3,3) - at(1,3) * at(3,0);
    const double A0213 = at(1,0) * at(3,2) - at(1,2) * at(3,0);
    const double A0312 = at(1,0) * at(2,3) - at(1,3) * at(2,0);
    const double A0212 = at(1,0) * at(2,2) - at(1,2) * at(2,0);
    const double A0113 = at(1,0) * at(3,1) - at(1,1) * at(3,0);
    const double A0112 = at(1,0) * at(2,1) - at(1,1) * at(2,0);

    double det_ = at(0,0) * ( at(1,1) * A2323 - at(1,2) * A1323 + at(1,3) * A1223 ) 
                - at(0,1) * ( at(1,0) * A2323 - at(1,2) * A0323 + at(1,3) * A0223 ) 
                + at(0,2) * ( at(1,0) * A1323 - at(1,1) * A0323 + at(1,3) * A0123 ) 
                - at(0,3) * ( at(1,0) * A1223 - at(1,1) * A0223 + at(1,2) * A0123 ) ;

    // inv det
    det_ = 1.0 / det_;

    Matrix_<double, 4, 4> ret;
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

template<typename DataT, unsigned int Rows, unsigned int Cols> 
template<unsigned int RowsNew, unsigned int ColsNew, unsigned int row, unsigned int col>
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, RowsNew, ColsNew> Matrix_<DataT, Rows, Cols>::copy_block() const
{
  Matrix_<DataT, RowsNew, ColsNew> ret;

  static_assert(row + RowsNew <= Rows);
  static_assert(col + ColsNew <= Cols);

  for(unsigned int i=0; i<RowsNew; i++)
  {
    for(unsigned int j=0; j<ColsNew; j++)
    {
      ret(i,j) = at(i+row, j+col);
    }
  }

  return ret;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
template<unsigned int RowsNew, unsigned int ColsNew>
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, RowsNew, ColsNew> Matrix_<DataT, Rows, Cols>::copy_block(unsigned int row, unsigned int col) const
{
  Matrix_<DataT, RowsNew, ColsNew> ret;

  for(unsigned int i=0; i<RowsNew; i++)
  {
    for(unsigned int j=0; j<ColsNew; j++)
    {
      ret(i,j) = at(i+row, j+col);
    }
  }

  return ret;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
template<unsigned int RowsBlock, unsigned int ColsBlock>
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::set_block(
  unsigned int row, 
  unsigned int col, 
  const Matrix_<DataT, RowsBlock, ColsBlock>& block)
{
  for(size_t i=0; i<RowsBlock; i++)
  {
    for(size_t j=0; j<ColsBlock; j++)
    {
      at(i+row, j+col) = block(i, j);
    }
  }
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
    static_assert(Rows == 3 && Cols == 3);
    *this = static_cast<Matrix_<DataT, Rows, Cols> >(q);
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
void Matrix_<DataT, Rows, Cols>::set(const EulerAngles_<DataT>& e)
{
    static_assert(Rows == 3);
    static_assert(Cols == 3);
    *this = static_cast<Matrix_<DataT, Rows, Cols> >(e);
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

/////
// CASTINGS
template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols>::operator Vector2_<DataT>() const 
{
    static_assert(Rows == 2 && Cols == 1);
    return {
        at(0, 0),
        at(1, 0)
    };
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols>::operator Vector3_<DataT>() const 
{
    static_assert(Rows == 3 && Cols == 1);
    return {
        at(0, 0),
        at(1, 0),
        at(2, 0)
    }; 
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols>::operator Quaternion_<DataT>() const 
{
    static_assert(Rows == 3 && Cols == 3);
    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    // TODO: test
    // 1. test: correct
    DataT tr = trace();

    Quaternion_<DataT> q;

    if (tr > 0) { 
        const DataT S = sqrtf(tr + 1.0) * 2; // S=4*qw 
        q.w = 0.25f * S;
        q.x = (at(2,1) - at(1,2)) / S;
        q.y = (at(0,2) - at(2,0)) / S; 
        q.z = (at(1,0) - at(0,1)) / S;
    } else if ((at(0,0) > at(1,1)) && (at(0,0) > at(2,2))) { 
        const DataT S = sqrtf(1.0 + at(0,0) - at(1,1) - at(2,2)) * 2.0; // S=4*qx 
        q.w = (at(2,1) - at(1,2)) / S;
        q.x = 0.25f * S;
        q.y = (at(0,1) + at(1,0)) / S; 
        q.z = (at(0,2) + at(2,0)) / S; 
    } else if (at(1,1) > at(2,2) ) { 
        const DataT S = sqrtf(1.0 + at(1,1) - at(0,0) - at(2,2)) * 2.0; // S=4*qy
        q.w = (at(0,2) - at(2,0)) / S;
        q.x = (at(0,1) + at(1,0)) / S; 
        q.y = 0.25f * S;
        q.z = (at(1,2) + at(2,1)) / S; 
    } else { 
        const DataT S = sqrtf(1.0 + at(2,2) - at(0,0) - at(1,1)) * 2.0; // S=4*qz
        q.w = (at(1,0) - at(0,1)) / S;
        q.x = (at(0,2) + at(2,0)) / S;
        q.y = (at(1,2) + at(2,1)) / S;
        q.z = 0.25 * S;
    }

    return q;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols>::operator EulerAngles_<DataT>() const 
{
    static_assert(Rows == 3 && Cols == 3);
    // extracted from knowledge of Matrix3x3::set(EulerAngles)
    // plus EulerAngles::set(Quaternion)
    // TODO: check. tested once: correct
    
    // roll (x-axis)
    const DataT sA_cB =  at(2,1);
    const DataT cA_cB =  at(2,2);
    
    // pitch (y-axis)
    const DataT sB    = -at(2,0);

    // yaw (z-axis)
    const DataT sC_cB =  at(1,0);
    const DataT cC_cB =  at(0,0);

    // roll (x-axis)
    EulerAngles_<DataT> e;
    e.roll = atan2(sA_cB, cA_cB);

    // pitch (y-axis)
    if (fabs(sB) >= 1.0)
    {
        e.pitch = copysignf(M_PI_2, sB); // use 90 degrees if out of range
    } else {
        e.pitch = asinf(sB);
    }

    // yaw (z-axis)
    e.yaw = atan2f(sC_cB, cC_cB);

    return e;
}

template<typename DataT, unsigned int Rows, unsigned int Cols> 
RMAGINE_INLINE_FUNCTION
Matrix_<DataT, Rows, Cols>::operator Transform_<DataT>() const 
{
    static_assert(Rows == 4 && Cols == 4);
    
    Transform_<DataT> T;
    T.set(*this);
    return T;
}



template<typename DataT, unsigned int Rows, unsigned int Cols> 
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
Matrix_<ConvT, Rows, Cols> Matrix_<DataT, Rows, Cols>::cast() const
{
    Matrix_<ConvT, Rows, Cols> res;

    for(unsigned int i=0; i<Rows; i++)
    {
        for(unsigned int j=0; j<Cols; j++)
        {
            res(i, j) = at(i, j);
        }
    }
    
    return res;
}



} // namespace rmagine