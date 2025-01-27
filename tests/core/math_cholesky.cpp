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

void precision_check()
{
  auto Cd = random_mat_for_chol<double, 6>();
  auto Cf = Cd.cast<float>();

  // std::cout << Cf << std::endl;

  rm::Matrix_<float, 6, 6> Lf;
  rm::chol(Cf, Lf);

  rm::Matrix_<double, 6, 6> Ld;
  rm::chol(Cd, Ld);

  // since C is symmetric positive-definite
  const rm::Matrix_<double, 6, 6> C2d = Ld*Ld.T();
  const rm::Matrix_<float, 6, 6> C2f = Lf*Lf.T();

  // compute error per element
  double errd = abssum(Cd - C2d) / static_cast<double>(6 * 6);
  float errf = abssum(Cf - C2f) / static_cast<float>(6 * 6);

  if(errf > ALLOWED_ERROR_CHOLESKY_SINGLE)
  {
    std::cout << "Error Float: " << errf << std::endl;
    RM_THROW(rm::Exception, "Error too high using single precision floating point.");
  }
  
  if(errd > ALLOWED_ERROR_CHOLESKY_DOUBLE)
  {
    std::cout << "Error Double: " << errd << std::endl;
    RM_THROW(rm::Exception, "Error too high using double precision floating point.");
  }
}

void test3()
{
  for(size_t i=0; i<10000; i++)
  {
    precision_check();
  }
}

void test1()
{
  std::cout << "---- TEST 1 ----" << std::endl; 
  auto C = random_mat_for_chol<float, 6>();

  rm::Matrix_<float, 6, 6> L;
  rm::chol(C, L);

  std::cout << C << std::endl;

  std::cout << L << std::endl;


  std::cout << "L*L'" << std::endl;

  std::cout << L * L.T() << std::endl;

  std::cout << "------" << std::endl;
}

void test2()
{
  // covariance sometimes contain zero elements on diagonal causing standard cholesky algorithm to generate NaNs:
  rm::Matrix6x6 C;
  C.setZeros();

  // this covariance, for examples, can be used to sample poses in 3D space on a 2D plane
  C(0,0) = 0.5; // x
  C(1,1) = 0.1; // y
  C(2,2) = 0.0; // z
  C(3,3) = 0.0; // roll
  C(4,4) = 0.0; // pitch
  C(5,5) = 0.0685389; // yaw

  // a correct solution exists:
  rm::Matrix6x6 Lgt;
  Lgt.setZeros();
  Lgt(0,0) = sqrt(C(0,0));
  Lgt(1,1) = sqrt(C(1,1));
  Lgt(2,2) = sqrt(C(2,2));
  Lgt(3,3) = sqrt(C(3,3));
  Lgt(4,4) = sqrt(C(4,4));
  Lgt(5,5) = sqrt(C(5,5));

  std::cout << "Ground Truth:" << std::endl;
  std::cout << Lgt << std::endl;

  std::cout << Lgt * Lgt.T() << std::endl;

  std::cout << "----" << std::endl;

  // To actually draw samples, the idea is to transform 
  // independently drawn samples from a standard normal
  // distribution N(mu=0, cov=I) and transform them
  // according to the given covariance 
  
  // This is done by using A*x + mu with A*A' = C. A is
  // - the transformation matrix
  // - can be determined using Cholesky decomposition

  rm::Matrix6x6 A;
  A.setZeros();
  rm::chol(C, A);
  std::cout << A << std::endl;

}

int main(int argc, char** argv)
{
  srand((unsigned int) time(0));

  std::cout << "Cholesky!" << std::endl;

  test1();

  test2();

  test3();
 

  return 0;
}

