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

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_CUH