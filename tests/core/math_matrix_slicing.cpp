


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
#include <utility>
#include <memory>


namespace rm = rmagine;



template<unsigned int Rows, unsigned int Cols>
rm::Matrix_<float, Rows, Cols> make_mat()
{
  rm::Matrix_<float, Rows, Cols> M;
  for(size_t i=0; i<Rows; i++)
  {
    for(size_t j=0; j<Cols; j++)
    {
      M(i,j) = i * Cols + j;
    }
  }
  return M;
}

void test_basics()
{
  std::cout << "-----------------------" << std::endl;
  std::cout << "1. BASICS" << std::endl;
  auto M1 = make_mat<6,6>();

  auto M2 = make_mat<10,10>();

  std::cout << "Size Check:" << std::endl;
  std::cout << "- " << sizeof(M1) <<  " == " << sizeof(float) * 6 * 6 << std::endl;
  std::cout << "- " << sizeof(M2) << " == " << sizeof(float) * 10 * 10 << std::endl;

  std::cout << "Slice:" << std::endl;
  auto M1s1 = M1.slice<3,3>(3,3);
  auto M2s1 = M2.slice<3,3>(5,0);
  auto M1s2 = M1.slice<3,3>(0,0);

  // copy assign
  std::cout << "Copy Assign:" << std::endl;
  M1s1 = M1s2;
  M2s1 = M1s2;

  std::cout << M1 << std::endl;
  std::cout << M2 << std::endl;

  std::cout << "Move Assign:" << std::endl;

  M1.slice<3,3>(3,0) = M2.slice<3,3>(7,7);

  std::cout << M1 << std::endl;
}


void test_const()
{
  std::cout << "-----------------------" << std::endl;
  std::cout << "2. CONST" << std::endl;
  rm::Matrix_<float, 10, 10> M = make_mat<10,10>();
  M.setZeros();
  const rm::Matrix_<float, 10, 10> M_const = make_mat<10,10>();

  auto Mbla = M_const.slice<3,3>(0,0);

  const rm::MatrixSlice_<float, 3, 3> Ms = M.slice<3,3>(0,0);
  
  // unconst
  auto Ms2 = Ms.slice<3,3>(0,0);

  std::cout << Ms2 << std::endl;

  std::cout << "move assign const slice to non-const memory" << std::endl;
  M.slice<3,3>(0,0) = M_const.slice<3,3>(5,5);

  std::cout << M << std::endl;

  // transposed
  std::cout << "transpose const slice and move assign it to non-const memory" << std::endl;
  M.slice<3,3>(3,3) = M_const.slice<3,3>(5,5).T();

  std::cout << M << std::endl;


  std::cout << "non-const Matrix<const float*>" << std::endl;

  rm::MatrixSlice_<const float, 3, 3> M_const_s = M_const.slice<3,3>(2,4); 

  // this gives a static error, as it is supposed to
  // M_const_s.transposeInplace();
}


void test_math()
{
  std::cout << "-----------------------" << std::endl;
  std::cout << "3. MATH" << std::endl;
  rm::Matrix_<float, 10, 10> M = make_mat<10,10>();
  M.setZeros();
  const rm::Matrix_<float, 10, 10> M_const = make_mat<10,10>();

  
  float test = M_const.slice<4,4>(0,0).slice<2,2>(0,0).det();
  std::cout << "Det: " << test << std::endl;

  std::cout << M_const.slice<2,2>(0,0).det() << std::endl;
  std::cout << M_const.slice<2,2>(2,0).det() << std::endl;
  std::cout << M_const.slice<2,2>(0,2).det() << std::endl;
  std::cout << M_const.slice<2,2>(2,2).det() << std::endl;
  
  std::cout << "---" << std::endl;

  rm::Matrix_<float, 4, 4> tl = M_const.slice<4,4>(1,0).T();

  std::cout << tl << std::endl;

  float test2 = M_const.slice<4,4>(1,0).T().det();
  std::cout << "Det: " << test2 << std::endl;
  std::cout << tl.det() << std::endl;

  M.slice<3,3>(3,0) += M_const.slice<3,3>(0,0).T();
  std::cout << M << std::endl;

  M.slice<3,3>(0,0) += M_const.slice<3,3>(0,0) + M_const.slice<3,3>(0,3);

  std::cout << M << std::endl;
}


int main(int argc, char** argv)
{
  test_basics();
  
  test_const();

  test_math();

  return 0;
}