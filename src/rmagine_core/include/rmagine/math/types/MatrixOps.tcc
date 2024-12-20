#include "MatrixOps.hpp"

#include "Matrix.hpp"

namespace rmagine
{

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::at(unsigned int row, unsigned int col)
{
  return static_cast<MatrixAccess*>(this)->access(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::at(unsigned int row, unsigned int col) volatile
{
  return static_cast<volatile MatrixAccess*>(this)->access(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::at(unsigned int row, unsigned int col) const
{
  return static_cast<const MatrixAccess*>(this)->access(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::at(unsigned int row, unsigned int col) volatile const
{
  return static_cast<volatile const MatrixAccess*>(this)->access(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator()(unsigned int row, unsigned int col)
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
volatile DataT& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator()(unsigned int row, unsigned int col) volatile
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator()(unsigned int row, unsigned int col) const
{
  return at(row, col);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
DataT MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator()(unsigned int row, unsigned int col) volatile const
{
  return at(row, col);
}

////////////////////
// setZeros
////////////////
template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION 
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setZeros()
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
void MatrixOps_<float, 3, 3, Matrix_>::setZeros()
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
void MatrixOps_<double, 3, 3, Matrix_>::setZeros()
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
void MatrixOps_<float, 4, 4, Matrix_>::setZeros()
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
void MatrixOps_<double, 4, 4, Matrix_>::setZeros()
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
template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION 
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setOnes()
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
void MatrixOps_<float, 3, 3, Matrix_>::setOnes()
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
void MatrixOps_<double, 3, 3, Matrix_>::setOnes()
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
void MatrixOps_<float, 4, 4, Matrix_>::setOnes()
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
void MatrixOps_<double, 4, 4, Matrix_>::setOnes()
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
template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setIdentity()
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
void MatrixOps_<float, 3, 3, Matrix_>::setIdentity()
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
void MatrixOps_<double, 3, 3, Matrix_>::setIdentity()
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
void MatrixOps_<float, 4, 4, Matrix_>::setIdentity()
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
void MatrixOps_<double, 4, 4, Matrix_>::setIdentity()
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


template< // class templates
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template< // function templates
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set(
  const OtherMatrixAccess_<std::add_const_t<DataT>, Rows, Cols>& other)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  
  for(size_t i = 0; i<Rows; i++)
  {
    for(size_t j = 0; j<Cols; j++)
    {
      at(i, j) = other(i, j);
    }
  }
}

template< // class
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template<
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set(
  const OtherMatrixAccess_<std::remove_const_t<DataT>, Rows, Cols>& other)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  
  for(size_t i = 0; i<Rows; i++)
  {
    for(size_t j = 0; j<Cols; j++)
    {
      at(i, j) = other(i, j);
    }
  }
}


template< // class templates
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template< // function templates
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
MatrixAccess_<DataT, Rows, Cols>& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator=(
  const OtherMatrixAccess_<std::add_const_t<DataT>, Rows, Cols>& other)
{
  set(other);
  return mat();
}

template< // class
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template<
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
MatrixAccess_<DataT, Rows, Cols>& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator=(
  const OtherMatrixAccess_<std::remove_const_t<DataT>, Rows, Cols>& other)
{
  set(other);
  return mat();
}

template< // class
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template<
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
MatrixAccess_<DataT, Rows, Cols>& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator=(
  OtherMatrixAccess_<std::add_const_t<DataT>, Rows, Cols>&& other)
{
  set(other);
  return mat();
}

template< // class
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
template<
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION
MatrixAccess_<DataT, Rows, Cols>& MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator=(
  OtherMatrixAccess_<std::remove_const_t<DataT>, Rows, Cols>&& other)
{
  set(other);
  return mat();
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> 
    MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::negate() const
{
    Matrix_<std::remove_const_t<DataT>, Rows, Cols> res;

    for(unsigned int i=0; i<Rows; i++)
    {
      for(unsigned int j=0; j<Cols; j++)
      {
        res(i, j) = -at(i, j);
      }
    }

    return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::negateInplace()
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  
  for(unsigned int i=0; i<Rows; i++)
  {
    for(unsigned int j=0; j<Cols; j++)
    {
      at(i, j) = -at(i, j);
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
template<unsigned int Cols2, 
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION 
Matrix_<std::remove_const_t<DataT>, Rows, Cols2> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::mult(
  const OtherMatrixAccess_<std::remove_const_t<DataT>, Cols, Cols2>& M) const
{
  // constexpr unsigned int Rows2 = Cols;
  constexpr unsigned int Rows3 = Rows;
  constexpr unsigned int Cols3 = Cols2;

  Matrix_<std::remove_const_t<DataT>, Rows3, Cols3> res;
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

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
template<unsigned int Cols2, 
  template<typename OtherMADataT, unsigned int OtherMARows, unsigned int OtherMACols> class OtherMatrixAccess_>
RMAGINE_INLINE_FUNCTION 
Matrix_<std::remove_const_t<DataT>, Rows, Cols2> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::mult(
  const OtherMatrixAccess_<std::add_const_t<DataT>, Cols, Cols2>& M) const
{
  // constexpr unsigned int Rows2 = Cols;
  constexpr unsigned int Rows3 = Rows;
  constexpr unsigned int Cols3 = Cols2;

  Matrix_<std::remove_const_t<DataT>, Rows3, Cols3> res;
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

  return res;
}



template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION 
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::multInplace(
  const Matrix_<DataT, Rows, Cols>& M)
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  static_assert(Rows == Cols, "Inplace mult is only allowed for square matrices");

  // tmp memory
  
  // TODO: test
  // - processing each column should be thread safe
  // #pragma omp parallel for
  for(unsigned int j = 0; j < Cols; j++)
  {
    // copy entire column
    DataT tmp[Rows];
    for(unsigned int i = 0; i < Rows; i++)
    {
      tmp[i] = at(i, j);
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

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::mult(
  const DataT& scalar) const
{
  Matrix_<DataTNonConst, Rows, Cols> res;

  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(i, j) = at(i, j) * scalar;
    }
  }

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::multInplace(
  const DataT& scalar)
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      at(i, j) *= scalar;
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::multEwise(
  const Matrix_<DataT, Rows, Cols>& M) const
{
  Matrix_<DataTNonConst, Rows, Cols> res;

  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(i, j) = at(i, j) * M(i,j);
    }
  }

  return res;
}


template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::div(
  const DataT& scalar) const
{
  Matrix_<DataTNonConst, Rows, Cols> res;

  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(i, j) = at(i, j) / scalar;
    }
  }

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::divInplace(const DataT& scalar)
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      at(i, j) /= scalar;
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION 
Vector3_<std::remove_const_t<DataT> > MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::mult(
  const Vector3_<std::remove_const_t<DataT> >& v) const
{
  if constexpr(Rows == 3 && Cols == 3)
  {
    return {
        at(0,0) * v.x + at(0,1) * v.y + at(0,2) * v.z, 
        at(1,0) * v.x + at(1,1) * v.y + at(1,2) * v.z, 
        at(2,0) * v.x + at(2,1) * v.y + at(2,2) * v.z
    };
  } 
  else if constexpr(Rows == 3 && Cols == 4
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

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION 
Vector2_<std::remove_const_t<DataT> > MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::mult(
  const Vector2_<std::remove_const_t<DataT> >& v) const
{
  if constexpr(Rows == 2 && Cols == 2)
  {
    return {
        at(0,0) * v.x + at(0,1) * v.y, 
        at(1,0) * v.x + at(1,1) * v.y,
    };
  } 
  else if constexpr(Rows == 2 && Cols == 3
              || Rows == 3 && Cols == 3)
  {
    return {
        at(0,0) * v.x + at(0,1) * v.y + at(0,2),
        at(1,0) * v.x + at(1,1) * v.y + at(1,2),
    };
  }
  
  return {NAN, NAN};
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::add(
  const Matrix_<DataT, Rows, Cols>& M) const
{
  Matrix_<DataTNonConst, Rows, Cols> res;
  
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(i, j) = at(i, j) + M(i, j);
    }
  }

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::addInplace(const Matrix_<DataT, Rows, Cols>& M)
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      at(i, j) += M(i, j);
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::addInplace(
  volatile Matrix_<DataT, Rows, Cols>& M) volatile
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      at(i, j) += M(i, j);
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::sub(
  const Matrix_<DataT, Rows, Cols>& M) const
{
  Matrix_<DataTNonConst, Rows, Cols> res;

  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(i, j) = at(i, j) - M(i, j);
    }
  }

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::subInplace(const Matrix_<DataT, Rows, Cols>& M)
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      at(i, j) -= M(i, j);
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Cols, Rows> 
    MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::transpose() const
{
  Matrix_<DataTNonConst, Cols, Rows> res;

  for(unsigned int i = 0; i < Rows; i++)
  {
    for(unsigned int j = 0; j < Cols; j++)
    {
      res(j, i) = at(i, j);
    }
  }

  return res;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::transposeInplace()
{
  static_assert(!std::is_const_v<DataT>, "Inplace operations are not allowed on const-typed matrices.");
  static_assert(Rows == Cols, "Inplace transpose is only allowed for square matrices");
  
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
void MatrixOps_<float, 3, 3, Matrix_>::transposeInplace()
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

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
std::remove_const_t<DataT> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::trace() const
{
  static_assert(Rows == Cols, "Trace only allowed on square matrices");

  DataTNonConst res = static_cast<DataTNonConst>(0);
  for(size_t i=0; i<Rows; i++)
  {
    res += at(i, i);
  }

  return res;
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
std::remove_const_t<DataT> mat_det(const MatrixAccess_<DataT, 2, 2>& mat)
{
  return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
std::remove_const_t<DataT> mat_det(const MatrixAccess_<DataT, 3, 3>& mat)
{
  return  mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) -
          mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) +
          mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
std::remove_const_t<DataT> mat_det(const MatrixAccess_<DataT, 4, 4>& mat)
{
  // TODO: check
  const float A2323 = mat(2,2) * mat(3,3) - mat(2,3) * mat(3,2);
  const float A1323 = mat(2,1) * mat(3,3) - mat(2,3) * mat(3,1);
  const float A1223 = mat(2,1) * mat(3,2) - mat(2,2) * mat(3,1);
  const float A0323 = mat(2,0) * mat(3,3) - mat(2,3) * mat(3,0);
  const float A0223 = mat(2,0) * mat(3,2) - mat(2,2) * mat(3,0);
  const float A0123 = mat(2,0) * mat(3,1) - mat(2,1) * mat(3,0);
  const float A2313 = mat(1,2) * mat(3,3) - mat(1,3) * mat(3,2);
  const float A1313 = mat(1,1) * mat(3,3) - mat(1,3) * mat(3,1);
  const float A1213 = mat(1,1) * mat(3,2) - mat(1,2) * mat(3,1);
  const float A2312 = mat(1,2) * mat(2,3) - mat(1,3) * mat(2,2);
  const float A1312 = mat(1,1) * mat(2,3) - mat(1,3) * mat(2,1);
  const float A1212 = mat(1,1) * mat(2,2) - mat(1,2) * mat(2,1);
  const float A0313 = mat(1,0) * mat(3,3) - mat(1,3) * mat(3,0);
  const float A0213 = mat(1,0) * mat(3,2) - mat(1,2) * mat(3,0);
  const float A0312 = mat(1,0) * mat(2,3) - mat(1,3) * mat(2,0);
  const float A0212 = mat(1,0) * mat(2,2) - mat(1,2) * mat(2,0);
  const float A0113 = mat(1,0) * mat(3,1) - mat(1,1) * mat(3,0);
  const float A0112 = mat(1,0) * mat(2,1) - mat(1,1) * mat(2,0);

  return  mat(0,0) * ( mat(1,1) * A2323 - mat(1,2) * A1323 + mat(1,3) * A1223 ) 
          - mat(0,1) * ( mat(1,0) * A2323 - mat(1,2) * A0323 + mat(1,3) * A0223 ) 
          + mat(0,2) * ( mat(1,0) * A1323 - mat(1,1) * A0323 + mat(1,3) * A0123 ) 
          - mat(0,3) * ( mat(1,0) * A1223 - mat(1,1) * A0223 + mat(1,2) * A0123 );
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
std::remove_const_t<DataT> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::det() const
{
  return mat_det(mat());
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, 2, 2> mat_inv(const MatrixAccess_<DataT, 2, 2>& mat)
{
  Matrix_<std::remove_const_t<DataT>, 2, 2> ret;
  
  std::add_const_t<DataT> invdet = 1.0f / mat_det(mat);
  ret(0, 0) =  mat(1, 1) * invdet;
  ret(0, 1) = -mat(0, 1) * invdet;
  ret(1, 0) = -mat(1, 0) * invdet;
  ret(1, 1) =  mat(0, 0) * invdet;
  
  return ret;
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, 3, 3> mat_inv(const MatrixAccess_<DataT, 3, 3>& mat)
{
  Matrix_<std::remove_const_t<DataT>, 3, 3> ret;

  std::add_const_t<DataT> invdet = 1.0f / mat_det(mat);

  ret(0, 0) = (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) * invdet;
  ret(0, 1) = (mat(0, 2) * mat(2, 1) - mat(0, 1) * mat(2, 2)) * invdet;
  ret(0, 2) = (mat(0, 1) * mat(1, 2) - mat(0, 2) * mat(1, 1)) * invdet;
  ret(1, 0) = (mat(1, 2) * mat(2, 0) - mat(1, 0) * mat(2, 2)) * invdet;
  ret(1, 1) = (mat(0, 0) * mat(2, 2) - mat(0, 2) * mat(2, 0)) * invdet;
  ret(1, 2) = (mat(1, 0) * mat(0, 2) - mat(0, 0) * mat(1, 2)) * invdet;
  ret(2, 0) = (mat(1, 0) * mat(2, 1) - mat(2, 0) * mat(1, 1)) * invdet;
  ret(2, 1) = (mat(2, 0) * mat(0, 1) - mat(0, 0) * mat(2, 1)) * invdet;
  ret(2, 2) = (mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1)) * invdet;

  return ret;
}

template<typename DataT, template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_>
RMAGINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, 4, 4> mat_inv(const MatrixAccess_<DataT, 4, 4>& mat)
{
  // https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
  // answer of willnode at Jun 8 '17 at 23:09

  std::add_const_t<DataT> A2323 = mat(2,2) * mat(3,3) - mat(2,3) * mat(3,2);
  std::add_const_t<DataT> A1323 = mat(2,1) * mat(3,3) - mat(2,3) * mat(3,1);
  std::add_const_t<DataT> A1223 = mat(2,1) * mat(3,2) - mat(2,2) * mat(3,1);
  std::add_const_t<DataT> A0323 = mat(2,0) * mat(3,3) - mat(2,3) * mat(3,0);
  std::add_const_t<DataT> A0223 = mat(2,0) * mat(3,2) - mat(2,2) * mat(3,0);
  std::add_const_t<DataT> A0123 = mat(2,0) * mat(3,1) - mat(2,1) * mat(3,0);
  std::add_const_t<DataT> A2313 = mat(1,2) * mat(3,3) - mat(1,3) * mat(3,2);
  std::add_const_t<DataT> A1313 = mat(1,1) * mat(3,3) - mat(1,3) * mat(3,1);
  std::add_const_t<DataT> A1213 = mat(1,1) * mat(3,2) - mat(1,2) * mat(3,1);
  std::add_const_t<DataT> A2312 = mat(1,2) * mat(2,3) - mat(1,3) * mat(2,2);
  std::add_const_t<DataT> A1312 = mat(1,1) * mat(2,3) - mat(1,3) * mat(2,1);
  std::add_const_t<DataT> A1212 = mat(1,1) * mat(2,2) - mat(1,2) * mat(2,1);
  std::add_const_t<DataT> A0313 = mat(1,0) * mat(3,3) - mat(1,3) * mat(3,0);
  std::add_const_t<DataT> A0213 = mat(1,0) * mat(3,2) - mat(1,2) * mat(3,0);
  std::add_const_t<DataT> A0312 = mat(1,0) * mat(2,3) - mat(1,3) * mat(2,0);
  std::add_const_t<DataT> A0212 = mat(1,0) * mat(2,2) - mat(1,2) * mat(2,0);
  std::add_const_t<DataT> A0113 = mat(1,0) * mat(3,1) - mat(1,1) * mat(3,0);
  std::add_const_t<DataT> A0112 = mat(1,0) * mat(2,1) - mat(1,1) * mat(2,0);

  std::remove_const_t<DataT> det_  = mat(0,0) * ( mat(1,1) * A2323 - mat(1,2) * A1323 + mat(1,3) * A1223 ) 
              - mat(0,1) * ( mat(1,0) * A2323 - mat(1,2) * A0323 + mat(1,3) * A0223 ) 
              + mat(0,2) * ( mat(1,0) * A1323 - mat(1,1) * A0323 + mat(1,3) * A0123 ) 
              - mat(0,3) * ( mat(1,0) * A1223 - mat(1,1) * A0223 + mat(1,2) * A0123 ) ;

  // inv det
  det_ = 1.0f / det_;

  Matrix_<std::remove_const_t<DataT>, 4, 4> ret;
  ret(0,0) = det_ *   ( mat(1,1) * A2323 - mat(1,2) * A1323 + mat(1,3) * A1223 );
  ret(0,1) = det_ * - ( mat(0,1) * A2323 - mat(0,2) * A1323 + mat(0,3) * A1223 );
  ret(0,2) = det_ *   ( mat(0,1) * A2313 - mat(0,2) * A1313 + mat(0,3) * A1213 );
  ret(0,3) = det_ * - ( mat(0,1) * A2312 - mat(0,2) * A1312 + mat(0,3) * A1212 );
  ret(1,0) = det_ * - ( mat(1,0) * A2323 - mat(1,2) * A0323 + mat(1,3) * A0223 );
  ret(1,1) = det_ *   ( mat(0,0) * A2323 - mat(0,2) * A0323 + mat(0,3) * A0223 );
  ret(1,2) = det_ * - ( mat(0,0) * A2313 - mat(0,2) * A0313 + mat(0,3) * A0213 );
  ret(1,3) = det_ *   ( mat(0,0) * A2312 - mat(0,2) * A0312 + mat(0,3) * A0212 );
  ret(2,0) = det_ *   ( mat(1,0) * A1323 - mat(1,1) * A0323 + mat(1,3) * A0123 );
  ret(2,1) = det_ * - ( mat(0,0) * A1323 - mat(0,1) * A0323 + mat(0,3) * A0123 );
  ret(2,2) = det_ *   ( mat(0,0) * A1313 - mat(0,1) * A0313 + mat(0,3) * A0113 );
  ret(2,3) = det_ * - ( mat(0,0) * A1312 - mat(0,1) * A0312 + mat(0,3) * A0112 );
  ret(3,0) = det_ * - ( mat(1,0) * A1223 - mat(1,1) * A0223 + mat(1,2) * A0123 );
  ret(3,1) = det_ *   ( mat(0,0) * A1223 - mat(0,1) * A0223 + mat(0,2) * A0123 );
  ret(3,2) = det_ * - ( mat(0,0) * A1213 - mat(0,1) * A0213 + mat(0,2) * A0113 );
  ret(3,3) = det_ *   ( mat(0,0) * A1212 - mat(0,1) * A0212 + mat(0,2) * A0112 );

  return ret;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Cols, Rows> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::inv() const
{
  return mat_inv(mat());
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
template<unsigned int RowsNew, unsigned int ColsNew>
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, RowsNew, ColsNew> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::copy_block(unsigned int row, unsigned int col) const
{
  Matrix_<DataTNonConst, RowsNew, ColsNew> ret;

  for(unsigned int i=0; i<RowsNew; i++)
  {
    for(unsigned int j=0; j<ColsNew; j++)
    {
      ret(i,j) = at(i+row, j+col);
    }
  }

  return ret;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
template<unsigned int RowsBlock, unsigned int ColsBlock>
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set_block(
  unsigned int row, 
  unsigned int col, 
  const Matrix_<DataT, RowsBlock, ColsBlock>& block)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  
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

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, Rows-1, Cols-1> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::rotation()
{
  static_assert(Rows == Cols);
  static_assert(Rows == 4);
  return mat_ptr()->template slice<Rows-1, Cols-1>(0,0);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
const MatrixSlice_<std::add_const_t<DataT>, Rows-1, Cols-1> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::rotation() const
{
  static_assert(Rows == Cols);
  static_assert(Rows == 4);
  return mat_ptr()->template slice<Rows-1, Cols-1>(0,0);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setRotation(const Matrix_<DataT, Rows-1, Cols-1>& R)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  
  for(unsigned int i=0; i < Rows - 1; i++)
  {
    for(unsigned int j=0; j < Cols - 1; j++)
    {
      at(i, j) = R(i, j);
    }
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setRotation(const Quaternion_<DataT>& q)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows >= 3 && Cols >= 3);

  Matrix_<DataTNonConst, 3, 3> R;
  R = q;
  setRotation(R);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setRotation(const EulerAngles_<DataT>& e)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows >= 3 && Cols >= 3);

  Matrix_<DataTNonConst, 3, 3> R;
  R = e;
  setRotation(R);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixSlice_<DataT, Rows-1, 1> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::translation()
{
  // TODO: check
  static_assert(Rows == Cols);
  static_assert(Rows == 4);
  return mat_ptr()->template slice<Rows-1, 1>(0, 2);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
const MatrixSlice_<std::add_const_t<DataT>, Rows-1, 1> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::translation() const
{
  // TODO: check
  static_assert(Rows == Cols);
  static_assert(Rows == 4);
  return mat_ptr()->template slice<Rows-1, 1>(0, 2);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setTranslation(const Matrix_<DataT, Rows-1, 1>& t)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  
  for(unsigned int i=0; i < Rows - 1; i++)
  {
    at(i, Cols-1) = t(i, 0);
  }
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setTranslation(const Vector2_<DataT>& t)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows >= 2 && Cols >= 3);

  at(0, Cols-1) = t.x;
  at(1, Cols-1) = t.y;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::setTranslation(const Vector3_<DataT>& t)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows >= 3 && Cols >= 4);

  at(0, Cols-1) = t.x;
  at(1, Cols-1) = t.y;
  at(2, Cols-1) = t.z;
}

template<
  typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
Matrix_<std::remove_const_t<DataT>, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::invRigid() const
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows == Cols);

  Matrix_<DataTNonConst, Rows, Cols> ret;
  ret.setIdentity();

  // TODO
  Matrix_<DataTNonConst, Rows-1, Cols-1> R = rotation();
  Matrix_<DataTNonConst, Rows-1, 1>      t = translation();

  R.transposeInplace();
  ret.setRotation(R);
  ret.setTranslation(- (R * t) );

  return ret;
}


template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set(const Quaternion_<DataT>& q)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows == 3 && Cols == 3);

  *this = static_cast<Matrix_<DataT, Rows, Cols> >(q);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set(const EulerAngles_<DataT>& e)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows == 3);
  static_assert(Cols == 3);

  *this = static_cast<Matrix_<DataT, Rows, Cols> >(e);
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
void MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::set(const Transform_<DataT>& T)
{
  static_assert(!std::is_const_v<DataT>, "Set operations are not allowed on const-typed matrices.");
  static_assert(Rows >= 3);
  static_assert(Cols >= 4);

  setIdentity();
  setRotation(T.R);
  setTranslation(T.t);
}

/////
// CASTINGS
template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator Vector2_<std::remove_const_t<DataT> >() const 
{
    static_assert(Rows == 2 && Cols == 1);
    return {
      at(0, 0),
      at(1, 0)
    };
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator Vector3_<std::remove_const_t<DataT> >() const 
{
    static_assert(Rows == 3 && Cols == 1);
    return {
      at(0, 0),
      at(1, 0),
      at(2, 0)
    }; 
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator Quaternion_<std::remove_const_t<DataT> >() const 
{
  static_assert(Rows == 3 && Cols == 3);
  // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  // TODO: test
  // 1. test: correct
  DataTConst tr = trace();

  Quaternion_<DataTNonConst> q;

  if (tr > 0) { 
    DataTConst S = sqrtf(tr + 1.0) * 2; // S=4*qw 
    q.w = 0.25f * S;
    q.x = (at(2,1) - at(1,2)) / S;
    q.y = (at(0,2) - at(2,0)) / S; 
    q.z = (at(1,0) - at(0,1)) / S;
  } else if ((at(0,0) > at(1,1)) && (at(0,0) > at(2,2))) { 
    DataTConst S = sqrtf(1.0 + at(0,0) - at(1,1) - at(2,2)) * 2.0; // S=4*qx 
    q.w = (at(2,1) - at(1,2)) / S;
    q.x = 0.25f * S;
    q.y = (at(0,1) + at(1,0)) / S; 
    q.z = (at(0,2) + at(2,0)) / S; 
  } else if (at(1,1) > at(2,2) ) { 
    DataTConst S = sqrtf(1.0 + at(1,1) - at(0,0) - at(2,2)) * 2.0; // S=4*qy
    q.w = (at(0,2) - at(2,0)) / S;
    q.x = (at(0,1) + at(1,0)) / S; 
    q.y = 0.25f * S;
    q.z = (at(1,2) + at(2,1)) / S;
  } else { 
    DataTConst S = sqrtf(1.0 + at(2,2) - at(0,0) - at(1,1)) * 2.0; // S=4*qz
    q.w = (at(1,0) - at(0,1)) / S;
    q.x = (at(0,2) + at(2,0)) / S;
    q.y = (at(1,2) + at(2,1)) / S;
    q.z = 0.25 * S;
  }

  return q;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator EulerAngles_<std::remove_const_t<DataT> >() const 
{
  static_assert(Rows == 3 && Cols == 3);
  // extracted from knowledge of Matrix3x3::set(EulerAngles)
  // plus EulerAngles::set(Quaternion)
  // TODO: check. tested once: correct
  
  // roll (x-axis)
  DataTConst sA_cB =  at(2,1);
  DataTConst cA_cB =  at(2,2);
  
  // pitch (y-axis)
  DataTConst sB    = -at(2,0);

  // yaw (z-axis)
  DataTConst sC_cB =  at(1,0);
  DataTConst cC_cB =  at(0,0);

  // roll (x-axis)
  EulerAngles_<DataTNonConst> e;
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

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
RMAGINE_INLINE_FUNCTION
MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::operator Transform_<std::remove_const_t<DataT> >() const 
{
  static_assert(Rows == 4 && Cols == 4);
  
  Transform_<DataTNonConst> T;
  T.set(*this);
  return T;
}

template<typename DataT, unsigned int Rows, unsigned int Cols, 
  template<typename MADataT, unsigned int MARows, unsigned int MACols> class MatrixAccess_> 
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
Matrix_<ConvT, Rows, Cols> MatrixOps_<DataT, Rows, Cols, MatrixAccess_>::cast() const
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