#include "rmagine/math/math.cuh"
#include <rmagine/math/math.h>
#include <rmagine/math/types.h>

namespace rmagine 
{

////////
// Generic Kernel
///

template<typename In1T, typename In2T, typename ResT>
__global__ void multNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] * B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void multNx1_kernel(
    const In1T* A,
    const In2T* b,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] * b[0];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void mult1xN_kernel(
    const In1T* a,
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = a[0] * B[id];
    }
}


////////////
// #multNxN
void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Quaternion, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNxN(A, B, C);
    return C;
}

void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), c.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> c(A.size());
    multNxN(A, b, c);
    return c;
}

/// TRANSFORM
void multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(T1.raw(), T2.raw(), Tr.raw(), T1.size());
}

Memory<Transform, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNxN(T1,T2,Tr);
    return Tr;
}

void multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(T.size());
    multNxN(T,x,c);
    return c;
}

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(M1.raw(), M2.raw(), Mr.raw(), M1.size());
}

Memory<Matrix3x3, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNxN(M1,M2,Mr);
    return Mr;
}

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), c.raw(), M.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNxN(M, x, c);
    return c;
}


////////
// #multNx1
///
void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& b,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
}

Memory<Quaternion, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& b)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(T1.raw(), t2.raw(), Tr.raw(), T1.size());
}

Memory<Transform, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNx1(T1,t2,Tr);
    return Tr;
}

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> C(T.size());
    multNx1(T,x,C);
    return C;
}

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(M1.raw(), m2.raw(), Mr.raw(), M1.size());
}

Memory<Matrix3x3, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNx1(M1,m2,Mr);
    return Mr;
}

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), C.raw(), M.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNx1(M, x, c);
    return c;
}

/////////////
// #mult1xN
////////
void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
}

Memory<Quaternion, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a, 
    const Memory<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B, 
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T2.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(t1.raw(), T2.raw(), Tr.raw(), T2.size());
}

Memory<Transform, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T2.size());
    mult1xN(t1, T2, Tr);
    return Tr;
}

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(t.raw(), X.raw(), C.raw(), X.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(t, X, C);
    return C;
}

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M2.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(m1.raw(), M2.raw(), Mr.raw(), M2.size());
}

Memory<Matrix3x3, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M2.size());
    mult1xN(m1, M2, Mr);
    return Mr;
}

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(m.raw(), X.raw(), C.raw(), X.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(m, X, C);
    return C;
}

} // namespace rmagine