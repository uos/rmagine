#ifndef RMAGINE_MATH_MATH_CUH
#define RMAGINE_MATH_MATH_CUH

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/math_batched.cuh>

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

inline Memory<Quaternion, VRAM_CUDA> operator*(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& B)
{
    return multNxN(A, B);
}

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

//////
// #divNxN
void divNxN(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> divNxN(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator/(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    return divNxN(A, B);
}

void divNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B, 
    Memory<Matrix3x3, VRAM_CUDA>& C);

Memory<Matrix3x3, VRAM_CUDA> divNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

inline Memory<Matrix3x3, VRAM_CUDA> operator/(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    return divNxN(A, B);
}

///////
// #divNxNIgnoreZeros
void divNxNIgnoreZeros(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

void divNxNIgnoreZeros(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Matrix3x3, VRAM_CUDA>& C);

Memory<Matrix3x3, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

void divNxNIgnoreZeros(
    const Memory<float, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<float, VRAM_CUDA>& C);

Memory<float, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<float, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

////////
// #divNxNInplace
void divNxNInplace(
    Memory<Vector, VRAM_CUDA>& A, 
    const Memory<float, VRAM_CUDA>& B);

void divNxNInplace(
    Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B);

////////
// #divNx1Inplace
void divNx1Inplace(
    Memory<Matrix3x3, VRAM_CUDA>& A, 
    const unsigned int& B);

////////
// #convert
void convert(const Memory<uint8_t, VRAM_CUDA>& from, 
    Memory<float, VRAM_CUDA>& to);

void convert(const Memory<bool, VRAM_CUDA>& from, 
    Memory<unsigned int, VRAM_CUDA>& to);

void convert(const Memory<unsigned int, VRAM_CUDA>& from, 
    Memory<bool, VRAM_CUDA>& to);

////////
// #pack
void pack(
    const Memory<Matrix3x3, VRAM_CUDA>& R,
    const Memory<Vector, VRAM_CUDA>& t,
    Memory<Transform, VRAM_CUDA>& T);

void pack(
    const Memory<Quaternion, VRAM_CUDA>& R,
    const Memory<Vector, VRAM_CUDA>& t,
    Memory<Transform, VRAM_CUDA>& T);


////////
// #multNxNTransposed

void multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& Cs);

Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2);

void multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& mask,
    Memory<Matrix3x3, VRAM_CUDA>& Cs);
    
Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& mask);

} // namespace rmagine

#endif // RMAGINE_MATH_MATH_CUH