#include <rmagine/math/math.h>

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
    for(size_t i=0; i<A.size(); i++)
    {
        C[i] = A[0] * B[i];
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


////////
// #mean

void mean2(const Memory<Vector, RAM>& X, Memory<Vector, RAM>& m)
{
    m[0] = mean(X);
}

Memory<Vector, RAM> mean2(const Memory<Vector, RAM>& X)
{
    Memory<Vector, RAM> m(1);
    mean2(X, m);
    return m;
}

} // namespace rmagine