#include "rmagine/math/memory_math.h"

#include "rmagine/util/prints.h"

#include <cassert>

#include <rmagine/math/linalg.h>

#include <tbb/parallel_for.h>

// BLOCK SIZE USED FOR THE WHOLE FILE
#define TBB_BLOCK_SIZE 512

namespace rmagine {


template<typename In1T, typename In2T, typename ResT>
void multNxN_generic(
  const MemoryView<In1T, RAM>& A,
  const MemoryView<In2T, RAM>& B,
  MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] * B[i];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void multNx1_generic(
    const MemoryView<In1T, RAM>& A,
    const MemoryView<In2T, RAM>& b,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] * b[0];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void mult1xN_generic(
    const MemoryView<In1T, RAM>& a,
    const MemoryView<In2T, RAM>& B,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for(tbb::blocked_range<size_t>(0, B.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = a[0] * B[i];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void addNxN_generic(
    const MemoryView<In1T, RAM>& A,
    const MemoryView<In2T, RAM>& B,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] + B[i];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void subNxN_generic(
    const MemoryView<In1T, RAM>& A,
    const MemoryView<In2T, RAM>& B,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] - B[i];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void subNx1_generic(
    const MemoryView<In1T, RAM>& A,
    const MemoryView<In2T, RAM>& b,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] - b[0];
    }
  });
}

template<typename In1T, typename In2T, typename ResT>
void sub1xN_generic(
    const MemoryView<In1T, RAM>& a,
    const MemoryView<In2T, RAM>& B,
    MemoryView<ResT, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, B.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = a[0] - B[i];
    }
  });
}

template<typename T>
void transpose_generic(
    const MemoryView<T, RAM>& A,
    MemoryView<T, RAM>& B)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      B[i] = A[i].transpose();
    }
  });
}

template<typename T>
void invert_generic(
    const MemoryView<T, RAM>& A,
    MemoryView<T, RAM>& B)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      B[i] = A[i].inv();
    }
  });
}

/////////////
// #multNxN
////////
void multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Quaternion, RAM>& B,
    MemoryView<Quaternion, RAM>& C)
{
  multNxN_generic(A, B, C);
}

Memory<Quaternion, RAM> multNxN(
    const MemoryView<Quaternion, RAM>& A, 
    const MemoryView<Quaternion, RAM>& B)
{
  Memory<Quaternion, RAM> C(A.size());
  multNxN(A, B, C);
  return C;
}

void multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b, 
    MemoryView<Vector, RAM>& c)
{
  multNxN_generic(A, b, c);
}

Memory<Vector, RAM> multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b)
{
  Memory<Vector, RAM> C(A.size());
  multNxN(A, b, C);
  return C;
}

void multNxN(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& T2,
    MemoryView<Transform, RAM>& Tr)
{
  multNxN_generic(T1, T2, Tr);
}

Memory<Transform, RAM> multNxN(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& T2)
{
  Memory<Transform, RAM> Tr(T1.size());
  multNxN(T1, T2, Tr);
  return Tr;
}

void multNxN(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& c)
{
  multNxN_generic(T, x, c);
}

Memory<Vector, RAM> multNxN(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x)
{
  Memory<Vector, RAM> C(T.size());
  multNxN(T, x, C);
  return C;
}

void multNxN(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& M2,
    MemoryView<Matrix3x3, RAM>& Mr)
{
  multNxN_generic(M1, M2, Mr);
}

Memory<Matrix3x3, RAM> multNxN(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& M2)
{
    Memory<Matrix3x3, RAM> Mr(M1.size());
    multNxN(M1, M2, Mr);
    return Mr;
}

void multNxN(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& c)
{
  multNxN_generic(M, x, c);
}

Memory<Vector, RAM> multNxN(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x)
{
  Memory<Vector, RAM> C(M.size());
  multNxN(M, x, C);
  return C;
}

/////////////
// #multNx1
////////
void multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Quaternion, RAM>& b,
    MemoryView<Quaternion, RAM>& C)
{
  multNx1_generic(A, b, C);
}

Memory<Quaternion, RAM> multNx1(
    const MemoryView<Quaternion, RAM>& A, 
    const MemoryView<Quaternion, RAM>& b)
{
    Memory<Quaternion, RAM> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b, 
    MemoryView<Vector, RAM>& C)
{
  multNx1_generic(A, b, C);
}

Memory<Vector, RAM> multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b)
{
  Memory<Vector, RAM> C(A.size());
  multNx1(A, b, C);
  return C;
}

void multNx1(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& t2,
    MemoryView<Transform, RAM>& Tr)
{
  multNx1_generic(T1, t2, Tr);
}

Memory<Transform, RAM> multNx1(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& t2)
{
  Memory<Transform, RAM> Tr(T1.size());
  multNx1(T1, t2, Tr);
  return Tr;
}

void multNx1(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& C)
{
  multNx1_generic(T, x, C);
}

Memory<Vector, RAM> multNx1(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x)
{
  Memory<Vector, RAM> C(T.size());
  multNx1(T, x, C);
  return C;
}

void multNx1(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& m2,
    MemoryView<Matrix3x3, RAM>& Mr)
{
  multNx1_generic(M1, m2, Mr);
}

Memory<Matrix3x3, RAM> multNx1(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& m2)
{
  Memory<Matrix3x3, RAM> Mr(M1.size());
  multNx1(M1, m2, Mr);
  return Mr;
}

void multNx1(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& C)
{
  multNx1_generic(M, x, C);
}

Memory<Vector, RAM> multNx1(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x)
{
  Memory<Vector, RAM> C(M.size());
  multNx1(M, x, C);
  return C;
}

/////////////
// #mult1xN
////////
void mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Quaternion, RAM>& B,
    MemoryView<Quaternion, RAM>& C)
{
  mult1xN_generic(a, B, C);
}

Memory<Quaternion, RAM> mult1xN(
    const MemoryView<Quaternion, RAM>& a, 
    const MemoryView<Quaternion, RAM>& B)
{
  Memory<Quaternion, RAM> C(B.size());
  mult1xN(a, B, C);
  return C;
}

void mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Vector, RAM>& B, 
    MemoryView<Vector, RAM>& C)
{
  mult1xN_generic(a, B, C);
}

Memory<Vector, RAM> mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Vector, RAM>& B)
{
  Memory<Vector, RAM> C(B.size());
  mult1xN(a, B, C);
  return C;
}

void mult1xN(
    const MemoryView<Transform, RAM>& t1,
    const MemoryView<Transform, RAM>& T2,
    MemoryView<Transform, RAM>& Tr)
{
  mult1xN_generic(t1, T2, Tr);
}

Memory<Transform, RAM> mult1xN(
    const MemoryView<Transform, RAM>& t1,
    const MemoryView<Transform, RAM>& T2)
{
  Memory<Transform, RAM> Tr(T2.size());
  mult1xN(t1, T2, Tr);
  return Tr;
}

void mult1xN(
    const MemoryView<Transform, RAM>& t,
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& C)
{
  mult1xN_generic(t, X, C);
}

Memory<Vector, RAM> mult1xN(
    const MemoryView<Transform, RAM>& t,
    const MemoryView<Vector, RAM>& X)
{
  Memory<Vector, RAM> C(X.size());
  mult1xN(t, X, C);
  return C;
}

void mult1xN(
    const MemoryView<Matrix3x3, RAM>& m1,
    const MemoryView<Matrix3x3, RAM>& M2,
    MemoryView<Matrix3x3, RAM>& Mr)
{
  mult1xN_generic(m1, M2, Mr);
}

Memory<Matrix3x3, RAM> mult1xN(
    const MemoryView<Matrix3x3, RAM>& m1,
    const MemoryView<Matrix3x3, RAM>& M2)
{
  Memory<Matrix3x3, RAM> Mr(M2.size());
  mult1xN(m1, M2, Mr);
  return Mr;
}

void mult1xN(
    const MemoryView<Matrix3x3, RAM>& m,
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& C)
{
  mult1xN_generic(m, X, C);
}

Memory<Vector, RAM> mult1xN(
    const MemoryView<Matrix3x3, RAM>& m,
    const MemoryView<Vector, RAM>& X)
{
  Memory<Vector, RAM> C(X.size());
  mult1xN(m, X, C);
  return C;
}


///////
// #add
void addNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B,
    MemoryView<Vector, RAM>& C)
{
  addNxN_generic(A, B, C);
}

Memory<Vector, RAM> addNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B)
{
  Memory<Vector, RAM> C(A.size());
  addNxN(A, B, C);
  return C;
}

////////
// #sub
void subNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B,
    MemoryView<Vector, RAM>& C)
{
  assert(A.size() == B.size());
  subNxN_generic(A, B, C);
}

Memory<Vector, RAM> subNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B)
{
  Memory<Vector, RAM> C(A.size());
  subNxN(A, B, C);
  return C;
}

void subNx1(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& b,
    MemoryView<Vector, RAM>& C)
{
  subNx1_generic(A, b, C);
}

Memory<Vector, RAM> subNx1(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& b)
{
  Memory<Vector, RAM> C(A.size());
  subNx1(A, b, C);
  return C;
}

void sub(
    const MemoryView<Vector, RAM>& A,
    const Vector& b,
    MemoryView<Vector, RAM>& C)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, A.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      C[i] = A[i] - b;
    }
  });
}

Memory<Vector, RAM> sub(
    const MemoryView<Vector, RAM>& A,
    const Vector& b)
{
  Memory<Vector, RAM> C(A.size());
  sub(A, b, C);
  return C;
}

/////
// #transpose
void transpose(
    const MemoryView<Matrix3x3, RAM>& A, 
    MemoryView<Matrix3x3, RAM>& B)
{
  transpose_generic(A, B);
}

Memory<Matrix3x3, RAM> transpose(
    const MemoryView<Matrix3x3, RAM>& A)
{
  Memory<Matrix3x3, RAM> B(A.size());
  transpose(A, B);
  return B;
}

void transpose(
    const MemoryView<Matrix4x4, RAM>& A,
    MemoryView<Matrix4x4, RAM>& B)
{
  transpose_generic(A, B);
}

Memory<Matrix4x4, RAM> transpose(
    const MemoryView<Matrix4x4, RAM>& A)
{
  Memory<Matrix4x4, RAM> B(A.size());
  transpose(A, B);
  return B;
}

//////
// #invert
void invert(
    const MemoryView<Matrix3x3, RAM>& A, 
    MemoryView<Matrix3x3, RAM>& B)
{
  invert_generic(A, B);
}

Memory<Matrix3x3, RAM> invert(
    const MemoryView<Matrix3x3, RAM>& A)
{
  Memory<Matrix3x3, RAM> B(A.size());
  invert(A, B);
  return B;
}

void invert(
    const MemoryView<Matrix4x4, RAM>& A,
    MemoryView<Matrix4x4, RAM>& B)
{
  invert_generic(A, B);
}

Memory<Matrix4x4, RAM> invert(
    const MemoryView<Matrix4x4, RAM>& A)
{
  Memory<Matrix4x4, RAM> B(A.size());
  invert(A, B);
  return B;
}

void invert(
    const MemoryView<Transform, RAM>& A,
    MemoryView<Transform, RAM>& B)
{
  invert_generic(A, B);
}

Memory<Transform, RAM> invert(
    const MemoryView<Transform, RAM>& A)
{
  Memory<Transform, RAM> B(A.size());
  invert(A, B);
  return B;
}

////////
// #pack
void pack(
    const MemoryView<Matrix3x3, RAM>& R,
    const MemoryView<Vector, RAM>& t,
    MemoryView<Transform, RAM>& T)
{
  for(unsigned int i=0; i<R.size(); i++)
  {
    T[i].R.set(R[i]);
    T[i].t = t[i];
  }
}

void pack(
    const MemoryView<Quaternion, RAM>& R,
    const MemoryView<Vector, RAM>& t,
    MemoryView<Transform, RAM>& T)
{
  for(unsigned int i=0; i<R.size(); i++)
  {
    T[i].R = R[i];
    T[i].t = t[i];
  }
}

////////
// #sum, #mean 
void sum(
    const MemoryView<Vector, RAM>& X, 
    MemoryView<Vector, RAM>& res)
{
  Vector s = {0, 0, 0};
  for(unsigned int i=0; i<X.size(); i++)
  {
    s += X[i];
  }
  res[0] = s;
}

Memory<Vector, RAM> sum(
    const MemoryView<Vector, RAM>& X)
{
  Memory<Vector, RAM> res(1);
  sum(X, res);
  return res;
}

void mean(
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& res)
{
  sum(X, res);
  res[0] /= static_cast<float>(X.size());
}

Memory<Vector, RAM> mean(
    const MemoryView<Vector, RAM>& X)
{
  Memory<Vector, RAM> res(1);
  mean(X, res);
  return res;
}

///////
// #cov   C = (v1 * v2.T) / N
void cov(
    const MemoryView<Vector, RAM>& v1,
    const MemoryView<Vector, RAM>& v2,
    MemoryView<Matrix3x3, RAM>& C)
{
  Matrix3x3 S;
  S.setZeros();
  
  for(unsigned int i=0; i<v1.size(); i++)
  {
    const Vector& a = v1[i];
    const Vector& b = v2[i];
    S(0,0) += a.x * b.x;
    S(1,0) += a.x * b.y;
    S(2,0) += a.x * b.z;
    S(0,1) += a.y * b.x;
    S(1,1) += a.y * b.y;
    S(2,1) += a.y * b.z;
    S(0,2) += a.z * b.x;
    S(1,2) += a.z * b.y;
    S(2,2) += a.z * b.z;
  }

  C[0] = S / static_cast<float>(v1.size());
}

Memory<Matrix3x3, RAM> cov(
    const MemoryView<Vector, RAM>& v1,
    const MemoryView<Vector, RAM>& v2)
{
  Memory<Matrix3x3, RAM> C(1);
  cov(v1, v2, C);
  return C;
}

/**
 * @brief decompose A = UWV* using singular value decomposition
 */
void svd(
    const MemoryView<Matrix3x3, RAM>& As,
    MemoryView<Matrix3x3, RAM>& Us,
    MemoryView<Matrix3x3, RAM>& Ws,
    MemoryView<Matrix3x3, RAM>& Vs)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, As.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      svd(As[i], Us[i], Ws[i], Vs[i]);
    }
  });
}

/**
 * @brief decompose A = UWV* using singular value decomposition
 * 
 * w is a vector which is the diagonal of matrix W
 */
void svd(
    const MemoryView<Matrix3x3, RAM>& As,
    MemoryView<Matrix3x3, RAM>& Us,
    MemoryView<Vector3, RAM>& ws,
    MemoryView<Matrix3x3, RAM>& Vs)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, As.size()),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      svd(As[i], Us[i], ws[i], Vs[i]);
    }
  });
}


void umeyama_transform(
    MemoryView<Transform, RAM>& Ts,
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs,
    const MemoryView<unsigned int, RAM>& n_meas)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, Cs.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      Ts[i] = umeyama_transform(ds[i], ms[i], Cs[i], n_meas[i]);
    }
  });
}

Memory<Transform, RAM> umeyama_transform(
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs,
    const MemoryView<unsigned int, RAM>& n_meas)
{
    Memory<Transform, RAM> ret(ds.size());
    umeyama_transform(ret, ds, ms, Cs, n_meas);
    return ret;
}

void umeyama_transform(
    MemoryView<Transform, RAM>& Ts,
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs)
{
  tbb::parallel_for( tbb::blocked_range<size_t>(0, Cs.size(), TBB_BLOCK_SIZE),
                       [&](const tbb::blocked_range<size_t>& r)
  {
    for(size_t i=r.begin(); i<r.end(); ++i)
    {
      Ts[i] = umeyama_transform(ds[i], ms[i], Cs[i]);
    }
  });
}

Memory<Transform, RAM> umeyama_transform(
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs)
{
    Memory<Transform, RAM> ret(ds.size());
    umeyama_transform(ret, ds, ms, Cs);
    return ret;
}


} // namespace rmagine