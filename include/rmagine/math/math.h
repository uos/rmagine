#ifndef RMAGINE_MATH_MATH_H
#define RMAGINE_MATH_MATH_H

#include <rmagine/math/types.h>
#include <rmagine/types/Memory.hpp>
#include <vector>

namespace rmagine
{

/////////////
// #multNxN
////////
void multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Quaternion, RAM>& B,
    Memory<Quaternion, RAM>& C);

Memory<Quaternion, RAM> multNxN(
    const Memory<Quaternion, RAM>& A, 
    const Memory<Quaternion, RAM>& B);

void multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b, 
    Memory<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b);

void multNxN(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& T2,
    Memory<Transform, RAM>& Tr);

Memory<Transform, RAM> multNxN(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& T2);

void multNxN(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x);

void multNxN(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& M2,
    Memory<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> multNxN(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& M2);

void multNxN(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x);

/////////////
// #multNx1
////////
void multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Quaternion, RAM>& b,
    Memory<Quaternion, RAM>& C);

Memory<Quaternion, RAM> multNx1(
    const Memory<Quaternion, RAM>& A, 
    const Memory<Quaternion, RAM>& b);

void multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b, 
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const Memory<Quaternion, RAM>& A,
    const Memory<Vector, RAM>& b);

void multNx1(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& t2,
    Memory<Transform, RAM>& Tr);

Memory<Transform, RAM> multNx1(
    const Memory<Transform, RAM>& T1,
    const Memory<Transform, RAM>& t2);

void multNx1(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const Memory<Transform, RAM>& T,
    const Memory<Vector, RAM>& x);

void multNx1(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& m2,
    Memory<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> multNx1(
    const Memory<Matrix3x3, RAM>& M1,
    const Memory<Matrix3x3, RAM>& m2);

void multNx1(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const Memory<Matrix3x3, RAM>& M,
    const Memory<Vector, RAM>& x);

/////////////
// #mult1xN
////////
void mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Quaternion, RAM>& B,
    Memory<Quaternion, RAM>& C);

Memory<Quaternion, RAM> mult1xN(
    const Memory<Quaternion, RAM>& a, 
    const Memory<Quaternion, RAM>& B);

void mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Vector, RAM>& B, 
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const Memory<Quaternion, RAM>& a,
    const Memory<Vector, RAM>& B);

void mult1xN(
    const Memory<Transform, RAM>& t1,
    const Memory<Transform, RAM>& T2,
    Memory<Transform, RAM>& Tr);

Memory<Transform, RAM> mult1xN(
    const Memory<Transform, RAM>& t1,
    const Memory<Transform, RAM>& T2);

void mult1xN(
    const Memory<Transform, RAM>& t,
    const Memory<Vector, RAM>& X,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const Memory<Transform, RAM>& t,
    const Memory<Vector, RAM>& X);

void mult1xN(
    const Memory<Matrix3x3, RAM>& m1,
    const Memory<Matrix3x3, RAM>& M2,
    Memory<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> mult1xN(
    const Memory<Matrix3x3, RAM>& m1,
    const Memory<Matrix3x3, RAM>& M2);

void mult1xN(
    const Memory<Matrix3x3, RAM>& m,
    const Memory<Vector, RAM>& X,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const Memory<Matrix3x3, RAM>& m,
    const Memory<Vector, RAM>& X);

//////
// #add
void addNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> addNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B);

inline Memory<Vector, RAM> operator+(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B)
{
    return addNxN(A, B);
}

//////
// #sub
void subNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> subNxN(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B);

inline Memory<Vector, RAM> operator-(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& B)
{
    return subNxN(A, B);
}

void subNx1(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& b,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> subNx1(
    const Memory<Vector, RAM>& A,
    const Memory<Vector, RAM>& b);

void sub(
    const Memory<Vector, RAM>& A,
    const Vector& b,
    Memory<Vector, RAM>& C);

Memory<Vector, RAM> sub(
    const Memory<Vector, RAM>& A,
    const Vector& b);

inline Memory<Vector, RAM> operator-(
    const Memory<Vector, RAM>& A,
    const Vector& b)
{
    return sub(A, b);
}

/////
// #transpose
void transpose(
    const Memory<Matrix3x3, RAM>& A, 
    Memory<Matrix3x3, RAM>& B);

Memory<Matrix3x3, RAM> transpose(
    const Memory<Matrix3x3, RAM>& A);

void transpose(
    const Memory<Matrix4x4, RAM>& A,
    Memory<Matrix4x4, RAM>& B);

Memory<Matrix4x4, RAM> transpose(
    const Memory<Matrix4x4, RAM>& A);

//////
// #invert
void invert(
    const Memory<Matrix3x3, RAM>& A, 
    Memory<Matrix3x3, RAM>& B);

Memory<Matrix3x3, RAM> invert(
    const Memory<Matrix3x3, RAM>& A);

void invert(
    const Memory<Matrix4x4, RAM>& A,
    Memory<Matrix4x4, RAM>& B);

Memory<Matrix4x4, RAM> invert(
    const Memory<Matrix4x4, RAM>& A);

void invert(
    const Memory<Transform, RAM>& A,
    Memory<Transform, RAM>& B);

Memory<Transform, RAM> invert(
    const Memory<Transform, RAM>& A);

////////
// #pack
void pack(
    const Memory<Matrix3x3, RAM>& R,
    const Memory<Vector, RAM>& t,
    Memory<Transform, RAM>& T);

void pack(
    const Memory<Quaternion, RAM>& R,
    const Memory<Vector, RAM>& t,
    Memory<Transform, RAM>& T);

///////
// #sum
void sum(
    const Memory<Vector, RAM>& X, 
    Memory<Vector, RAM>& res);

Memory<Vector, RAM> sum(
    const Memory<Vector, RAM>& X);

//////
// #mean
void mean(
    const Memory<Vector, RAM>& X,
    Memory<Vector, RAM>& res);

Memory<Vector,RAM> mean(
    const Memory<Vector, RAM>& X);

///////
// #cov   C = (v1 * v2.T) / N
void cov(
    const Memory<Vector, RAM>& v1,
    const Memory<Vector, RAM>& v2,
    Memory<Matrix3x3, RAM>& C
);

Memory<Matrix3x3, RAM> cov(
    const Memory<Vector, RAM>& v1,
    const Memory<Vector, RAM>& v2
);




} // namespace rmagine

#endif // RMAGINE_MATH_MATH_H