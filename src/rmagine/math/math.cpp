#include "rmagine/math/math.h"

#include "rmagine/util/prints.h"

#include <cassert>

namespace rmagine {


template<typename In1T, typename In2T, typename ResT>
void multNxN_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] * B[i];
    }
}

template<typename In1T, typename In2T, typename ResT>
void multNx1_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] * B[0];
    }
}



template<typename In1T, typename In2T, typename ResT>
void mult1xN_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<B.size(); i++)
    {
        C[i] = A[0] * B[i];
    }
}

template<typename In1T, typename In2T, typename ResT>
void addNxN_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] + B[i];
    }
}

template<typename In1T, typename In2T, typename ResT>
void subNxN_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] - B[i];
    }
}

template<typename In1T, typename In2T, typename ResT>
void subNx1_generic(
    const Memory<In1T, RAM>& A,
    const Memory<In2T, RAM>& b,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] - b[0];
    }
}

template<typename In1T, typename In2T, typename ResT>
void sub1xN_generic(
    const Memory<In1T, RAM>& a,
    const Memory<In2T, RAM>& B,
    Memory<ResT, RAM>& C)
{
    #pragma omp parallel for
    for(size_t i=0; i<B.size(); i++)
    {
        C[i] = a[0] - B[i];
    }
}

template<typename T>
void transpose_generic(
    const Memory<T, RAM>& A,
    Memory<T, RAM>& B)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        B[i] = A[i].transpose();
    }
}

template<typename T>
void invert_generic(
    const Memory<T, RAM>& A,
    Memory<T, RAM>& B)
{
    #pragma omp parallel for
    for(size_t i=0; i<A.size(); i++)
    {
        B[i] = A[i].inv();
    }
}

/////////////
// #multNxN
////////
void multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Quaternion, RAM>& B,
    Memory<Quaternion, RAM>& C)
{
    multNxN_generic(A, B, C);
}

Memory<Quaternion, RAM> multNxN(
    const Memory<Quaternion, RAM>& A, 
    const Memory<Quaternion, RAM>& B)
{
    Memory<Quaternion, RAM> C(A.size());
    multNxN(A, B, C);
    return C;
}

void multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b, 
    Memory<Vector, RAM>& c)
{
    multNxN_generic(A, b, c);
}

Memory<Vector, RAM> multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b)
{
    Memory<Vector, RAM> C(A.size());
    multNxN(A, b, C);
    return C;
}

void multNxN(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& T2,
    Memory<Transform, RAM>& Tr)
{
    multNxN_generic(T1, T2, Tr);
}

Memory<Transform, RAM> multNxN(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& T2)
{
    Memory<Transform, RAM> Tr(T1.size());
    multNxN(T1, T2, Tr);
    return Tr;
}

void multNxN(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& c)
{
    multNxN_generic(T, x, c);
}

Memory<Vector, RAM> multNxN(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x)
{
    Memory<Vector, RAM> C(T.size());
    multNxN(T, x, C);
    return C;
}

void multNxN(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& M2,
    Memory<Matrix3x3, RAM>& Mr)
{
    multNxN_generic(M1, M2, Mr);
}

Memory<Matrix3x3, RAM> multNxN(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& M2)
{
    Memory<Matrix3x3, RAM> Mr(M1.size());
    multNxN(M1, M2, Mr);
    return Mr;
}

void multNxN(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& c)
{
    multNxN_generic(M, x, c);
}

Memory<Vector, RAM> multNxN(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x)
{
    Memory<Vector, RAM> C(M.size());
    multNxN(M, x, C);
    return C;
}

/////////////
// #multNx1
////////
void multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Quaternion, RAM>& b,
    Memory<Quaternion, RAM>& C)
{
    multNx1_generic(A, b, C);
}

Memory<Quaternion, RAM> multNx1(
    const Memory<Quaternion, RAM>& A, 
    const Memory<Quaternion, RAM>& b)
{
    Memory<Quaternion, RAM> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b, 
    Memory<Vector, RAM>& C)
{
    multNx1_generic(A, b, C);
}

Memory<Vector, RAM> multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b)
{
    Memory<Vector, RAM> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& t2,
    Memory<Transform, RAM>& Tr)
{
    multNx1_generic(T1, t2, Tr);
}

Memory<Transform, RAM> multNx1(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& t2)
{
    Memory<Transform, RAM> Tr(T1.size());
    multNx1(T1, t2, Tr);
    return Tr;
}

void multNx1(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& C)
{
    multNx1_generic(T, x, C);
}

Memory<Vector, RAM> multNx1(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x)
{
    Memory<Vector, RAM> C(T.size());
    multNx1(T, x, C);
    return C;
}

void multNx1(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& m2,
    Memory<Matrix3x3, RAM>& Mr)
{
    multNx1_generic(M1, m2, Mr);
}

Memory<Matrix3x3, RAM> multNx1(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& m2)
{
    Memory<Matrix3x3, RAM> Mr(M1.size());
    multNx1(M1, m2, Mr);
    return Mr;
}

void multNx1(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& C)
{
    multNx1_generic(M, x, C);
}

Memory<Vector, RAM> multNx1(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x)
{
    Memory<Vector, RAM> C(M.size());
    multNx1(M, x, C);
    return C;
}

/////////////
// #mult1xN
////////
void mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Quaternion, RAM>& B,
    Memory<Quaternion, RAM>& C)
{
    mult1xN_generic(a, B, C);
}

Memory<Quaternion, RAM> mult1xN(
    const Memory<Quaternion, RAM>& a, 
    const Memory<Quaternion, RAM>& B)
{
    Memory<Quaternion, RAM> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Vector, RAM>& B, 
    Memory<Vector, RAM>& C)
{
    mult1xN_generic(a, B, C);
}

Memory<Vector, RAM> mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Vector, RAM>& B)
{
    Memory<Vector, RAM> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Transform, RAM>& t1,
    const Memory<Transform, RAM>& T2,
    Memory<Transform, RAM>& Tr)
{
    mult1xN_generic(t1, T2, Tr);
}

Memory<Transform, RAM> mult1xN(
    const Memory<Transform, RAM>& t1,
    const Memory<Transform, RAM>& T2)
{
    Memory<Transform, RAM> Tr(T2.size());
    mult1xN(t1, T2, Tr);
    return Tr;
}

void mult1xN(
    const Memory<Transform, RAM>& t,
    const Memory<Vector, RAM>& X,
    Memory<Vector, RAM>& C)
{
    mult1xN_generic(t, X, C);
}

Memory<Vector, RAM> mult1xN(
    const Memory<Transform, RAM>& t,
    const Memory<Vector, RAM>& X)
{
    Memory<Vector, RAM> C(X.size());
    mult1xN(t, X, C);
    return C;
}

void mult1xN(
    const Memory<Matrix3x3, RAM>& m1,
    const Memory<Matrix3x3, RAM>& M2,
    Memory<Matrix3x3, RAM>& Mr)
{
    mult1xN_generic(m1, M2, Mr);
}

Memory<Matrix3x3, RAM> mult1xN(
    const Memory<Matrix3x3, RAM>& m1,
    const Memory<Matrix3x3, RAM>& M2)
{
    Memory<Matrix3x3, RAM> Mr(M2.size());
    mult1xN(m1, M2, Mr);
    return Mr;
}

void mult1xN(
    const Memory<Matrix3x3, RAM>& m,
    const Memory<Vector, RAM>& X,
    Memory<Vector, RAM>& C)
{
    mult1xN_generic(m, X, C);
}

Memory<Vector, RAM> mult1xN(
    const Memory<Matrix3x3, RAM>& m,
    const Memory<Vector, RAM>& X)
{
    Memory<Vector, RAM> C(X.size());
    mult1xN(m, X, C);
    return C;
}


///////
// #add
void addNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B,
    Memory<Vector, RAM>& C)
{
    addNxN_generic(A, B, C);
}

Memory<Vector, RAM> addNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B)
{
    Memory<Vector, RAM> C(A.size());
    addNxN(A, B, C);
    return C;
}

////////
// #sub
void subNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B,
    Memory<Vector, RAM>& C)
{
    assert(A.size() == B.size());
    subNxN_generic(A, B, C);
}

Memory<Vector, RAM> subNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B)
{
    Memory<Vector, RAM> C(A.size());
    subNxN(A, B, C);
    return C;
}

void subNx1(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& b,
    Memory<Vector, RAM>& C)
{
    subNx1_generic(A, b, C);
}

Memory<Vector, RAM> subNx1(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& b)
{
    Memory<Vector, RAM> C(A.size());
    subNx1(A, b, C);
    return C;
}

void sub(
    const Memory<Vector, RAM>& A,
    const Vector& b,
    Memory<Vector, RAM>& C)
{
    #pragma omp parralel for
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[i] - b;
    }
}

Memory<Vector, RAM> sub(
    const Memory<Vector, RAM>& A,
    const Vector& b)
{
    Memory<Vector, RAM> C(A.size());
    sub(A, b, C);
    return C;
}

/////
// #transpose
void transpose(
    const Memory<Matrix3x3, RAM>& A, 
    Memory<Matrix3x3, RAM>& B)
{
    transpose_generic(A, B);
}

Memory<Matrix3x3, RAM> transpose(
    const Memory<Matrix3x3, RAM>& A)
{
    Memory<Matrix3x3, RAM> B(A.size());
    transpose(A, B);
    return B;
}

void transpose(
    const Memory<Matrix4x4, RAM>& A,
    Memory<Matrix4x4, RAM>& B)
{
    transpose_generic(A, B);
}

Memory<Matrix4x4, RAM> transpose(
    const Memory<Matrix4x4, RAM>& A)
{
    Memory<Matrix4x4, RAM> B(A.size());
    transpose(A, B);
    return B;
}

//////
// #invert
void invert(
    const Memory<Matrix3x3, RAM>& A, 
    Memory<Matrix3x3, RAM>& B)
{
    invert_generic(A, B);
}

Memory<Matrix3x3, RAM> invert(
    const Memory<Matrix3x3, RAM>& A)
{
    Memory<Matrix3x3, RAM> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const Memory<Matrix4x4, RAM>& A,
    Memory<Matrix4x4, RAM>& B)
{
    invert_generic(A, B);
}

Memory<Matrix4x4, RAM> invert(
    const Memory<Matrix4x4, RAM>& A)
{
    Memory<Matrix4x4, RAM> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const Memory<Transform, RAM>& A,
    Memory<Transform, RAM>& B)
{
    invert_generic(A, B);
}

Memory<Transform, RAM> invert(
    const Memory<Transform, RAM>& A)
{
    Memory<Transform, RAM> B(A.size());
    invert(A, B);
    return B;
}

////////
// #pack
void pack(
    const Memory<Matrix3x3, RAM>& R,
    const Memory<Vector, RAM>& t,
    Memory<Transform, RAM>& T)
{
    for(unsigned int i=0; i<R.size(); i++)
    {
        T[i].R.set(R[i]);
        T[i].t = t[i];
    }
}

void pack(
    const Memory<Quaternion, RAM>& R,
    const Memory<Vector, RAM>& t,
    Memory<Transform, RAM>& T)
{
    for(unsigned int i=0; i<R.size(); i++)
    {
        T[i].R = R[i];
        T[i].t = t[i];
    }
}

////////
// #sum, #mean 
void sum(const Memory<Vector, RAM>& X, Memory<Vector, RAM>& res)
{
    Vector s = {0, 0, 0};
    for(unsigned int i=0; i<X.size(); i++)
    {
        s += X[i];
    }
    res[0] = s;
}

Memory<Vector, RAM> sum(const Memory<Vector, RAM>& X)
{
    Memory<Vector, RAM> res(1);
    sum(X, res);
    return res;
}

void mean(const Memory<Vector, RAM>& X, Memory<Vector, RAM>& res)
{
    sum(X, res);
    res[0] /= static_cast<float>(X.size());
}

Memory<Vector, RAM> mean(const Memory<Vector, RAM>& X)
{
    Memory<Vector, RAM> res(1);
    mean(X, res);
    return res;
}

///////
// #cov   C = (v1 * v2.T) / N
void cov(
    const Memory<Vector, RAM>& v1,
    const Memory<Vector, RAM>& v2,
    Memory<Matrix3x3, RAM>& C)
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
    const Memory<Vector, RAM>& v1,
    const Memory<Vector, RAM>& v2)
{
    Memory<Matrix3x3, RAM> C(1);
    cov(v1, v2, C);
    return C;
}


} // namespace rmagine