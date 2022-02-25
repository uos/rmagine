#include "rmagine/math/math.cuh"
#include <rmagine/math/math.h>
#include <rmagine/math/types.h>

namespace rmagine 
{

__global__ void multNxN_kernel(
    const Quaternion* A,
    const Quaternion* B,
    Quaternion* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] * B[id];
    }
}

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

__global__ void multNxN_kernel(
    const Quaternion* A,
    const Vector* b,
    Vector* c,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        c[id] = A[id] * b[id];
    }
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

__global__ void multNxN_kernel(
    const Transform* T,
    const Vector* x,
    Vector* c,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        c[id] = T[id] * x[id];
    }
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

__global__ void multNxN_kernel(
    const Matrix3x3* M,
    const Vector* x,
    Vector* c,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        c[id] = M[id] * x[id];
    }
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

} // namespace rmagine