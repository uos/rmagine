


// #include "rmagine/math/types.h"

// #include "rmagine/math/types/EulerAngles.hpp"
#include "rmagine/math/types/Matrix.hpp"

// #include <rmagine/math/math.h>


// #include <rmagine/util/StopWatch.hpp>

// #include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>


// #include <cblas.h>

#include <stdint.h>
#include <string.h>
#include <vector>
#include <iostream>



namespace rm = rmagine;


template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
class MatrixSlice_;

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride = Rows>
class Matrix_;

template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride, 
  template<typename MaDataType, unsigned int MaRows, unsigned int MaCols, unsigned int MaStride> class MatrixAccess_>
class MatrixOps_;




template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
class MatrixSlice_
: public MatrixOps_<DataT, Rows, Cols, Stride, MatrixSlice_>
{
public:

  // using MatrixOps = MatrixOps_<MatrixData_<DataT, Rows, Cols, Stride> >;
  using DataType = DataT;

  MatrixSlice_() = delete;
  explicit MatrixSlice_(DataT* data, unsigned int row, unsigned int col)
  : data(data)
  , row_offset(row)
  , col_offset(col)
  {

  }

  inline DataT& at(unsigned int row, unsigned int col)
  {
    return data[(col + col_offset) * Stride + (row + row_offset)];
  }

  inline DataT at(unsigned int row, unsigned int col) const
  {
    return data[(col + col_offset) * Stride + (row + row_offset)];
  }

  template<unsigned int SliceRows, unsigned int SliceCols>
  MatrixSlice_<DataT, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col)
  {
    return MatrixSlice_<DataT, SliceRows, SliceCols, Stride>(data, row + row_offset, col + col_offset);
  }

private:
  DataT* data;
  const unsigned int row_offset;
  const unsigned int col_offset;
};



template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
class Matrix_ 
: public MatrixOps_<DataT, Rows, Cols, Stride, Matrix_> // inherit all functions via CRTP
{
public:

  // using MatrixOps = MatrixOps_<MatrixData_<DataT, Rows, Cols, Stride> >;
  using DataType = DataT;

  inline DataType& at(unsigned int row, unsigned int col)
  {
    return data[col * Stride + row];
  }

  inline DataType at(unsigned int row, unsigned int col) const
  {
    return data[col * Stride + row];
  }

  template<unsigned int SliceRows, unsigned int SliceCols>
  MatrixSlice_<DataType, SliceRows, SliceCols, Stride> slice(unsigned int row, unsigned int col)
  {
    return MatrixSlice_<DataType, SliceRows, SliceCols, Stride>(&data[0], row, col);
  }

private:
  DataT data[Cols * Stride];
};


template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride, 
  template<typename MaDataType, unsigned int MaRows, unsigned int MaCols, unsigned int MaStride> class MatrixAccess_>
class MatrixOps_
{
public:
  // using DataType = typename MatrixAccess::DataType;
  using MatrixAccess = MatrixAccess_<DataT, Rows, Cols, Stride>;

  void addOne()
  {
    data_.at(0,0) += 1.0;
  }
protected:
  MatrixAccess& data_ = static_cast<MatrixAccess&>(*this);
};


template<unsigned int Rows, unsigned int Cols>
Matrix_<float, Rows, Cols> make_mat_data()
{
  Matrix_<float, Rows, Cols> ret;

  for(size_t i=0; i<Rows; i++)
  {
    for(size_t j=0; j<Cols; j++)
    {
      ret.at(i,j) = static_cast<float>(i * Cols + j);
    }
  }

  return ret;
}

// template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
// std::ostream& operator<<(std::ostream& os, const Matrix_<DataT, Rows, Cols, Stride>& M)
// {
//   for(size_t i=0; i<Rows; i++)
//   {
//     for(size_t j=0; j<Cols; j++)
//     {
//       os << M.at(i, j) << " ";
//     }
//     os << "\n";
//   }
//   return os;
// }

// template<typename DataT, unsigned int Rows, unsigned int Cols, unsigned int Stride>
// std::ostream& operator<<(std::ostream& os, const MatrixSlice_<DataT, Rows, Cols, Stride>& M)
// {
//   for(size_t i=0; i<Rows; i++)
//   {
//     for(size_t j=0; j<Cols; j++)
//     {
//       os << M.at(i, j) << " ";
//     }
//     os << "\n";
//   }
//   return os;
// }

void test_basics()
{
  // Matrix_<float, 10, 10> M = make_mat();

  // auto M2 = M.slice<3,3>(3,3);
  // M2.at(0,0) = 4.0;

  // std::cout << M.at(3,3) << std::endl;
}

void test_const()
{
  // const Matrix_<float, 10, 10> M = make_mat();

  // auto M2 = M.slice<3,3>(3,3);
  // M2.at(0,0) = 4.0;

  // std::cout << M.at(3,3) << std::endl;
}


using Matrix3x3 = Matrix_<float, 3, 3>;

int main(int argc, char** argv)
{
  // std::cout << "Rmagine Test: Matrix Slice" << std::endl;


  rm::Matrix_<float, 6, 6> M;
  M.setZeros();
  M(1,1) = 2.0;
  

  std::cout << sizeof(M) <<  " == " << sizeof(float) * 6 * 6 << std::endl;


  auto br = M.slice<3,3>(3,3);
  br(0,0) = 25.0;

  rm::Matrix_<float, 3, 3> bla;
  bla.setIdentity();

  M.slice<3,3>(0,0).set(bla);

  std::cout << " -----------------" << std::endl;


  M.slice<3,3>(0,0).set(M.slice<3,3>(3,3));

  std::cout << M << std::endl;



  auto bl = M.slice<3,3>(3,0);

  bl = br;

  std::cout << M << std::endl;

  // this doesnt work. what is happening here?
  M.slice<3,3>(0,3) = M.slice<3,3>(3,3);

  // std::cout << M << std::endl;


  return 0;
}