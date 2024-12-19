#include "Matrix.hpp"

#include <iostream>

namespace rmagine {

//////////////////
// MatrixSlice

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
MatrixSlice_<DataT, Rows, Cols, Stride>::MatrixSlice_(DataT* data, const unsigned int row, const unsigned int col)
:data(data)
,row_offset(row)
,col_offset(col)
{
  
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT& MatrixSlice_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col)
{
  return data[(col + col_offset) * Stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixSlice_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) volatile
{
  return data[(col + col_offset) * Stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT MatrixSlice_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) const
{
  return data[(col + col_offset) * Stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT MatrixSlice_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) volatile const
{
  return data[(col + col_offset) * Stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, SliceRows, SliceCols, Stride> MatrixSlice_<DataT, Rows, Cols, Stride>::slice(unsigned int row, unsigned int col)
{
  return MatrixSlice_<DataT, SliceRows, SliceCols, Stride>(&data[0], row + row_offset, col + col_offset);
}


//////////////////
// Matrix

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT& Matrix_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col)
{
  return data[col * Stride + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
volatile DataT& Matrix_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) volatile
{
  return data[col * Stride + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) const
{
  return data[col * Stride + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols, Stride>::access(unsigned int row, unsigned int col) volatile const
{
  return data[col * Stride + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, SliceRows, SliceCols, Stride> Matrix_<DataT, Rows, Cols, Stride>::slice(unsigned int row, unsigned int col)
{
  return MatrixSlice_<DataT, SliceRows, SliceCols, Stride>(&data[0], row, col);
}

} // namespace rmagine