#include "Matrix.hpp"

#include <iostream>

namespace rmagine {

//////////////////
// MatrixSlice

template<typename DataT, unsigned int Rows, unsigned int Cols>
MatrixSlice_<DataT, Rows, Cols>::MatrixSlice_(
  DataT* data, 
  const unsigned int stride,
  const unsigned int row, 
  const unsigned int col)
:data(data)
,stride(stride)
,row_offset(row)
,col_offset(col)
{
  
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& MatrixSlice_<DataT, Rows, Cols>::access(
  unsigned int row, unsigned int col)
{
  return data[(col + col_offset) * stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixSlice_<DataT, Rows, Cols>::access(
  unsigned int row, unsigned int col) volatile
{
  return data[(col + col_offset) * stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixSlice_<DataT, Rows, Cols>::access(
  unsigned int row, unsigned int col) const
{
  return data[(col + col_offset) * stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT MatrixSlice_<DataT, Rows, Cols>::access(
  unsigned int row, unsigned int col) volatile const
{
  return data[(col + col_offset) * stride + (row + row_offset)];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, SliceRows, SliceCols> MatrixSlice_<DataT, Rows, Cols>::slice(
  unsigned int row, unsigned int col)
{
  return MatrixSlice_<DataT, SliceRows, SliceCols>(&data[0], stride, row + row_offset, col + col_offset);
}


template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
const MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols> MatrixSlice_<DataT, Rows, Cols>::slice(
  unsigned int row, unsigned int col) const
{
  return MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols>(
    data, stride, row + row_offset, col + col_offset);
}

//////////////////
// Matrix

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT& Matrix_<DataT, Rows, Cols>::access(unsigned int row, unsigned int col)
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
volatile DataT& Matrix_<DataT, Rows, Cols>::access(unsigned int row, unsigned int col) volatile
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::access(unsigned int row, unsigned int col) const
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
RMAGINE_INLINE_FUNCTION
DataT Matrix_<DataT, Rows, Cols>::access(unsigned int row, unsigned int col) volatile const
{
  return data[col * Rows + row];
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, SliceRows, SliceCols> Matrix_<DataT, Rows, Cols>::slice(
  unsigned int row, unsigned int col)
{
  return MatrixSlice_<DataT, SliceRows, SliceCols>(&data[0], Rows, row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
template<unsigned int SliceRows, unsigned int SliceCols>
RMAGINE_INLINE_FUNCTION
const MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols> Matrix_<DataT, Rows, Cols>::slice(
  unsigned int row, unsigned int col) const
{
  return MatrixSlice_<std::add_const_t<DataT>, SliceRows, SliceCols>(
    const_cast<DataT*>(&data[0]), Rows, row, col);
}

} // namespace rmagine