#include <iostream>
#include <random>

#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/linalg.h>

#include <Eigen/Dense>

#include <stdint.h>
#include <string.h>

namespace rm = rmagine;




template<typename DataT, unsigned int Rows, unsigned int Cols>
DataT abssum(
  const rm::Matrix_<DataT, Rows, Cols>& A)
{
  DataT res = 0;

  for(size_t i=0; i<Rows; i++)
  {
    for(size_t j=0; j<Cols; j++)
    {
      res += abs(A(i, j));
    }
  }
  
  return res;
}

template<typename DataT, unsigned int Dim>
rm::Matrix_<DataT, Dim, Dim> random_mat_for_chol()
{
  static std::mt19937 gen;
  static std::uniform_real_distribution<DataT> dist(-10.0, 10.0);

  rm::Matrix_<DataT, Dim, Dim> ret;
  for(size_t i=0; i<Dim; i++)
  {
    for(size_t j=0; j<Dim; j++)
    {
      ret(i,j) = dist(gen);
    }
  }
  // make it positive semidefinite
  return ret * ret.T();
}

#define ALLOWED_ERROR_CHOLESKY_SINGLE 0.00001
#define ALLOWED_ERROR_CHOLESKY_DOUBLE 0.00000001

void test1()
{
  auto Cd = random_mat_for_chol<double, 6>();
  auto Cf = Cd.cast<float>();

  // std::cout << Cf << std::endl;

  rm::Matrix_<float, 6, 6> Lf;
  rm::chol(Cf, Lf);

  rm::Matrix_<double, 6, 6> Ld;
  rm::chol(Cd, Ld);

  // std::cout << Ld << std::endl;

  // since C is symmetric positive-definite
  const rm::Matrix_<double, 6, 6> C2d = Ld*Ld.T();
  const rm::Matrix_<float, 6, 6> C2f = Lf*Lf.T();

  // compute error per element
  double errd = abssum(Cd - C2d) / static_cast<double>(6 * 6);
  float errf = abssum(Cf - C2f) / static_cast<float>(6 * 6);
  std::cout << "Error Double: " << errd << std::endl;
  std::cout << "Error Float: " << errf << std::endl;

  if(errf > ALLOWED_ERROR_CHOLESKY_SINGLE)
  {
    RM_THROW(rm::Exception, "Error too high using single precision floating point.");
  } 
  
  if(errd > ALLOWED_ERROR_CHOLESKY_DOUBLE)
  {
    RM_THROW(rm::Exception, "Error too high using double precision floating point.");
  }
}


int main(int argc, char** argv)
{
  srand((unsigned int) time(0));

  std::cout << "Cholesky!" << std::endl;

  for(size_t i=0; i<10000; i++)
  {
    test1();
  }
 

  return 0;
}

