#include <iostream>


#include "rmagine/math/types.h"

#include <rmagine/math/math.h>


#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <Eigen/Dense>

// #include <cblas.h>

#include <stdint.h>
#include <string.h>


namespace rm = rmagine;

template<typename DataT, unsigned int Rows, unsigned int Cols>
class MatrixData_;


template<typename DataT, unsigned int Rows, unsigned int Cols>
class MatrixData_ 
{
public:

  inline DataT& at(unsigned int row, unsigned int col)
  {
    return data[col * Rows + row];
  }

  inline DataT at(unsigned int row, unsigned int col) const
  {
    return data[col * Rows + row];
  }

  template<unsigned int SliceRows, unsigned int SliceCols>
  class Slice_;

  template<unsigned int SliceRows, unsigned int SliceCols>
  Slice_<SliceRows, SliceCols> slice(const unsigned int row, const unsigned int col)
  {
    return Slice_<SliceRows, SliceCols>(&data[0], row, col);
  }

  template<unsigned int SliceRows, unsigned int SliceCols>
  class Slice_ 
  {
    public:

      Slice_(DataT* data, const unsigned int row, const unsigned int col)
      :data(data)
      ,row_offset(row)
      ,col_offset(col)
      {
        
      }

      inline DataT& at(unsigned int row, unsigned int col)
      {
        return data[(col_offset + col)  * Rows + (row_offset + row)];
      }

      inline DataT at(unsigned int row, unsigned int col) const
      {
        return data[(col_offset + col)  * Rows + (row_offset + row)];
      }

      template<unsigned int SliceRowsNew, unsigned int SliceColsNew>
      RMAGINE_INLINE_FUNCTION
      Slice_<SliceRowsNew, SliceColsNew> slice(unsigned int row, unsigned int col)
      {
        return Slice_<SliceRowsNew, SliceColsNew>(data, row_offset + row, col_offset + col);
      }

    private:
      DataT* data;
      const unsigned int col_offset;
      const unsigned int row_offset;
  };

private:
  DataT data[Cols * Rows];
};

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename DataTI, unsigned int RowsI, unsigned int ColsI> class DataContainer_ = MatrixData_>
class Matrix_ 
: public DataContainer_<DataT, Rows, Cols>
{
public:
  // either slice or data
  using DataContainer = DataContainer_<DataT, Rows, Cols>;

  template<unsigned int SliceRows, unsigned int SliceCols>
  MatrixData_<DataT, Rows, Cols>::Slice_<SliceRows, SliceCols> slice(const unsigned int row, const unsigned int col)
  {
    return DataContainer::template slice<SliceRows, SliceCols>(row, col);
  }
};


Matrix_<float, 10, 10> make_mat()
{
  Matrix_<float, 10, 10> ret;

  for(size_t i=0; i<10; i++)
  {
    for(size_t j=0; j<10; j++)
    {
      ret.at(i,j) = static_cast<float>(i + j * 2.0);
    }
  }

  return ret;
}

void test_basics()
{
  Matrix_<float, 10, 10> M = make_mat();

  auto M2 = M.slice<3,3>(3,3);
  M2.at(0,0) = 4.0;

  std::cout << M.at(3,3) << std::endl;
}

void test_const()
{
  // const Matrix_<float, 10, 10> M = make_mat();

  // auto M2 = M.slice<3,3>(3,3);
  // M2.at(0,0) = 4.0;

  // std::cout << M.at(3,3) << std::endl;
}

int main(int argc, char** argv)
{
  std::cout << "Rmagine Test: Matrix Slice" << std::endl;

  test_basics();

  return 0;
}