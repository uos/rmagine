#ifndef RMAGINE_MATH_MATH_CUH
#define RMAGINE_MATH_MATH_CUH

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>

namespace rmagine 
{

/////////////
// #multNxN
////////
void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& B);

void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b);

void multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2);

void multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x);

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2);

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x);

/////////////
// #multNx1
////////
void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& b,
    Memory<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& b);

void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b);

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2,
    Memory<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2);

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x);

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2);

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x);

/////////////
// #mult1xN
////////
void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a, 
    const Memory<Quaternion, VRAM_CUDA>& B);

void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B, 
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B);

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2);

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X);

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2);

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X);


//////
// #add
void addNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> addNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator+(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B)
{
    return addNxN(A, B);
}

//////
// #sub
void subNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> subNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator-(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B)
{
    return subNxN(A, B);
}

/////
// #transpose
void transpose(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    Memory<Matrix3x3, VRAM_CUDA>& B);

Memory<Matrix3x3, VRAM_CUDA> transpose(
    const Memory<Matrix3x3, VRAM_CUDA>& A);

void transpose(
    const Memory<Matrix4x4, VRAM_CUDA>& A,
    Memory<Matrix4x4, VRAM_CUDA>& B);

Memory<Matrix4x4, VRAM_CUDA> transpose(
    const Memory<Matrix4x4, VRAM_CUDA>& A);


//////
// #invert
void invert(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    Memory<Matrix3x3, VRAM_CUDA>& B);

Memory<Matrix3x3, VRAM_CUDA> invert(
    const Memory<Matrix3x3, VRAM_CUDA>& A);

void invert(
    const Memory<Matrix4x4, VRAM_CUDA>& A,
    Memory<Matrix4x4, VRAM_CUDA>& B);

Memory<Matrix4x4, VRAM_CUDA> invert(
    const Memory<Matrix4x4, VRAM_CUDA>& A);

void invert(
    const Memory<Transform, VRAM_CUDA>& A,
    Memory<Transform, VRAM_CUDA>& B);

Memory<Transform, VRAM_CUDA> invert(
    const Memory<Transform, VRAM_CUDA>& A);

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_CUH