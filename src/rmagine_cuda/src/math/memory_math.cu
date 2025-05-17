#include "rmagine/math/memory_math.cuh"
#include "rmagine/math/types.h"
#include "rmagine/math/linalg.cuh"
#include "rmagine/util/cuda/CudaDebug.hpp"

namespace rmagine 
{

namespace cuda
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
__global__ void multNxN_conv_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id].set(A[id] * B[id]);
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


template<typename In1T, typename In2T, typename ResT>
__global__ void addNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] + B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void subNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] - B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void subNx1_kernel(
    const In1T* A,
    const In2T* b,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] - b[0];
    }
}


template<typename T>
__global__ void transpose_kernel(
    const T* A,
    T* B,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        B[id] = A[id].transpose();
    }
}

template<typename T>
__global__ void transposeInplace_kernel(
    T* A,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id].transposeInplace();
    }
}


template<typename T>
__global__ void invert_kernel(
    const T* A,
    T* B,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        B[id] = A[id].inv();
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void divNxN_kernel(
    const In1T* A, 
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] / B[id];
    }
}

template<typename ConvT, typename In1T, typename In2T, typename ResT>
__global__ void divNxN_conv_kernel(
    const In1T* A, 
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] / static_cast<ConvT>(B[id]);
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const Vector* A, 
    const unsigned int* B,
    Vector* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id].setZeros();
        }
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const Matrix3x3* A,
    const unsigned int* B,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id].setZeros();
        }
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const float* A, 
    const unsigned int* B,
    float* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id] = 0.0;
        }
    }
}

__global__ void divNxNInplace_kernel(
    Vector* A, 
    const float* B,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= B[id];
    }
}

__global__ void divNxNInplace_kernel(
    Matrix3x3* A, 
    const unsigned int* B,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= static_cast<float>(B[id]);
    }
}

template<typename T>
__global__ void divNx1Inplace_kernel(
    T* A,
    unsigned int b,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= static_cast<float>(b);
    }
}

__global__ void convert_kernel(const uint8_t* from, float* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = static_cast<float>(from[id]);
    }
}

__global__
void convert_kernel(const bool* from, unsigned int* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = static_cast<unsigned int>(from[id]);
    }
}

__global__
void convert_kernel(const unsigned int* from, bool* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = (from[id] > 0);
    }
}

__global__ void pack_kernel(
    const Matrix3x3* R, 
    const Vector* t, // Vector3d / Vector3f
    Transform* T,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        T[id].R.set(R[id]);
        T[id].t = t[id];
    }
}

__global__ void pack_kernel(
    const Quaternion* R, 
    const Vector* t, // Vector3d / Vector3f
    Transform* T,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        T[id].R = R[id];
        T[id].t = t[id];
    }
}

__global__ void covParts_kernel(
    const Vector* a, 
    const Vector* b,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id](0,0) = a[id].x * b[id].x;
        C[id](1,0) = a[id].x * b[id].y;
        C[id](2,0) = a[id].x * b[id].z;
        C[id](0,1) = a[id].y * b[id].x;
        C[id](1,1) = a[id].y * b[id].y;
        C[id](2,1) = a[id].y * b[id].z;
        C[id](0,2) = a[id].z * b[id].x;
        C[id](1,2) = a[id].z * b[id].y;
        C[id](2,2) = a[id].z * b[id].z;
    }
}

__global__ void covParts_kernel(
    const Vector* a, 
    const Vector* b,
    const bool* corr,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(corr[id])
        {
            C[id](0,0) = a[id].x * b[id].x;
            C[id](1,0) = a[id].x * b[id].y;
            C[id](2,0) = a[id].x * b[id].z;
            C[id](0,1) = a[id].y * b[id].x;
            C[id](1,1) = a[id].y * b[id].y;
            C[id](2,1) = a[id].y * b[id].z;
            C[id](0,2) = a[id].z * b[id].x;
            C[id](1,2) = a[id].z * b[id].y;
            C[id](2,2) = a[id].z * b[id].z;
        } else {
            C[id].setZeros();
        }
    }
}

template<unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if(blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if(blockSize >=  2) sdata[tid] += sdata[tid + 1];
}



__global__ void normalizeInplace_kernel(
    Quaternion* q,
    unsigned int N)
{
    // TODO: this was empty. test this
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        q[id].normalizeInplace();
    }
}

template<unsigned int blockSize>
__global__ void cov_kernel(
    const Vector* v1,
    const Vector* v2,
    Matrix3x3* res,
    unsigned int N)
{
    __shared__ Matrix3x3 sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    sdata[tid].setZeros();
    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < N)
        {
            const Vector& a = v1[globId + blockSize * i];
            const Vector& b = v2[globId + blockSize * i];
            sdata[tid](0,0) += a.x * b.x;
            sdata[tid](1,0) += a.x * b.y;
            sdata[tid](2,0) += a.x * b.z;
            sdata[tid](0,1) += a.y * b.x;
            sdata[tid](1,1) += a.y * b.y;
            sdata[tid](2,1) += a.y * b.z;
            sdata[tid](0,2) += a.z * b.x;
            sdata[tid](1,2) += a.z * b.y;
            sdata[tid](2,2) += a.z * b.z;
        }
    }
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0] / static_cast<float>(N);
    }
}

} // namespace cuda

////////////
// #multNxN
void multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Quaternion, VRAM_CUDA>& B,
    MemoryView<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Quaternion, VRAM_CUDA> multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A, 
    const MemoryView<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNxN(A, B, C);
    return C;
}

void multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b, 
    MemoryView<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), c.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> c(A.size());
    multNxN(A, b, c);
    return c;
}

/// TRANSFORM
void multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& T2,
    MemoryView<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(T1.raw(), T2.raw(), Tr.raw(), T1.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNxN(T1,T2,Tr);
    return Tr;
}

void multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(T.size());
    multNxN(T,x,c);
    return c;
}

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(M1.raw(), M2.raw(), Mr.raw(), M1.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNxN(M1,M2,Mr);
    return Mr;
}

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Quaternion, VRAM_CUDA>& Qres)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    cuda::multNxN_conv_kernel<<<gridSize, blockSize>>>(M1.raw(), M2.raw(), Qres.raw(), M1.size());
    RM_CUDA_DEBUG();
}

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), c.raw(), M.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNxN(M, x, c);
    return c;
}

////////
// #multNx1
///
void multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Quaternion, VRAM_CUDA>& b,
    MemoryView<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Quaternion, VRAM_CUDA> multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A, 
    const MemoryView<Quaternion, VRAM_CUDA>& b)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b, 
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& t2,
    MemoryView<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(T1.raw(), t2.raw(), Tr.raw(), T1.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& t2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNx1(T1,t2,Tr);
    return Tr;
}

void multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> C(T.size());
    multNx1(T,x,C);
    return C;
}

void multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& m2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(M1.raw(), m2.raw(), Mr.raw(), M1.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& m2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNx1(M1,m2,Mr);
    return Mr;
}

void multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), C.raw(), M.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNx1(M, x, c);
    return c;
}

void multNx1(
    const MemoryView<Matrix4x4, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    cuda::multNx1_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), C.raw(), M.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Matrix4x4, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNx1(M, x, c);
    return c;
}

/////////////
// #mult1xN
////////
void mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Quaternion, VRAM_CUDA>& B,
    MemoryView<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
    RM_CUDA_DEBUG();
}

Memory<Quaternion, VRAM_CUDA> mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a, 
    const MemoryView<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Vector, VRAM_CUDA>& B, 
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t1,
    const MemoryView<Transform, VRAM_CUDA>& T2,
    MemoryView<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T2.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(t1.raw(), T2.raw(), Tr.raw(), T2.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t1,
    const MemoryView<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T2.size());
    mult1xN(t1, T2, Tr);
    return Tr;
}

void mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(t.raw(), X.raw(), C.raw(), X.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t,
    const MemoryView<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(t, X, C);
    return C;
}

void mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M2.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(m1.raw(), M2.raw(), Mr.raw(), M2.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M2.size());
    mult1xN(m1, M2, Mr);
    return Mr;
}

void mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(m.raw(), X.raw(), C.raw(), X.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(m, X, C);
    return C;
}

void mult1xN(
    const MemoryView<Matrix4x4, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    cuda::mult1xN_kernel<<<gridSize, blockSize>>>(m.raw(), X.raw(), C.raw(), X.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix4x4, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(m, X, C);
    return C;
}

///////
// #add
void addNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::addNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> addNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    addNxN(A, B, C);
    return C;
}

void addNxN(
    const MemoryView<float, VRAM_CUDA>& A,
    const MemoryView<float, VRAM_CUDA>& B,
    MemoryView<float, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::addNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<float, VRAM_CUDA> addNxN(
    const MemoryView<float, VRAM_CUDA>& A,
    const MemoryView<float, VRAM_CUDA>& B)
{
    Memory<float, VRAM_CUDA> C(A.size());
    addNxN(A, B, C);
    return C;
}


////////
// #sub
void subNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::subNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> subNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    subNxN(A, B, C);
    return C;
}

void subNx1(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::subNx1_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> subNx1(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    subNx1(A, b, C);
    return C;
}

/////
// #transpose
void transpose(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    MemoryView<Matrix3x3, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::transpose_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> transpose(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A)
{
    Memory<Matrix3x3, VRAM_CUDA> B(A.size());
    transpose(A, B);
    return B;
}

void transpose(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A,
    MemoryView<Matrix4x4, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::transpose_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix4x4, VRAM_CUDA> transpose(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A)
{
    Memory<Matrix4x4, VRAM_CUDA> B(A.size());
    transpose(A, B);
    return B;
}

///////
// #transposeInplace
void transposeInplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::transposeInplace_kernel<<<gridSize, blockSize>>>(A.raw(), A.size());
    RM_CUDA_DEBUG();
}

//////
// #invert
void invert(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    MemoryView<Matrix3x3, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> invert(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A)
{
    Memory<Matrix3x3, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A,
    MemoryView<Matrix4x4, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix4x4, VRAM_CUDA> invert(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A)
{
    Memory<Matrix4x4, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const MemoryView<Transform, VRAM_CUDA>& A,
    MemoryView<Transform, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> invert(
    const MemoryView<Transform, VRAM_CUDA>& A)
{
    Memory<Transform, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

//////
// #divNxN
void divNxN(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxN_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> divNxN(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    divNxN(A, B, C);
    return C;
}

void divNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B, 
    MemoryView<Matrix3x3, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxN_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> divNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A,
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    Memory<Matrix3x3, VRAM_CUDA> C(A.size());
    divNxN(A, B, C);
    return C;
}

///////
// #divNxNIgnoreZeros
void divNxNIgnoreZeros(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

void divNxNIgnoreZeros(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Matrix3x3, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    Memory<Matrix3x3, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

void divNxNIgnoreZeros(
    const MemoryView<float, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<float, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
    RM_CUDA_DEBUG();
}

Memory<float, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<float, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    Memory<float, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

////////
// #divNxNInplace
void divNxNInplace(
    MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<float, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxNInplace_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
    RM_CUDA_DEBUG();
}

void divNxNInplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNxNInplace_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

////////
// #divNx1Inplace
void divNx1Inplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const unsigned int& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNx1Inplace_kernel<<<gridSize, blockSize>>>(A.raw(), B, A.size());
    RM_CUDA_DEBUG();
}

void divNx1Inplace(
    MemoryView<Vector, VRAM_CUDA>& A, 
    const unsigned int& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    cuda::divNx1Inplace_kernel<<<gridSize, blockSize>>>(A.raw(), B, A.size());
}


////////
// #convert
void convert(
    const MemoryView<uint8_t, VRAM_CUDA>& from, 
    MemoryView<float, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    cuda::convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
    RM_CUDA_DEBUG();
}

void convert(
    const MemoryView<bool, VRAM_CUDA>& from, 
    MemoryView<unsigned int, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    cuda::convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
}

void copy(const MemoryView<unsigned int, VRAM_CUDA>& from, 
    MemoryView<bool, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    cuda::convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
    RM_CUDA_DEBUG();
}

////////
// #pack
void pack(
    const MemoryView<Matrix3x3, VRAM_CUDA>& R,
    const MemoryView<Vector, VRAM_CUDA>& t,
    MemoryView<Transform, VRAM_CUDA>& T)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (R.size() + blockSize - 1) / blockSize;
    cuda::pack_kernel<<<gridSize, blockSize>>>(R.raw(), t.raw(), T.raw(), R.size());
}

void pack(
    const MemoryView<Quaternion, VRAM_CUDA>& R,
    const MemoryView<Vector, VRAM_CUDA>& t,
    MemoryView<Transform, VRAM_CUDA>& T)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (R.size() + blockSize - 1) / blockSize;
    cuda::pack_kernel<<<gridSize, blockSize>>>(R.raw(), t.raw(), T.raw(), R.size());
    RM_CUDA_DEBUG();
}

////////
// #multNxNTransposed
void multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Cs)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (m1.size() + blockSize - 1) / blockSize;
    cuda::covParts_kernel<<<gridSize, blockSize>>>(m1.raw(), m2.raw(), Cs.raw(), m1.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2)
{
    Memory<Matrix3x3, VRAM_CUDA> Cs(m1.size());
    multNxNTransposed(m1, m2, Cs);
    return Cs;
}

void multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& mask,
    MemoryView<Matrix3x3, VRAM_CUDA>& Cs)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (m1.size() + blockSize - 1) / blockSize;
    cuda::covParts_kernel<<<gridSize, blockSize>>>(m1.raw(), m2.raw(), mask.raw(), Cs.raw(), m1.size());
    RM_CUDA_DEBUG();
}
    
Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& mask)
{
    Memory<Matrix3x3, VRAM_CUDA> Cs(m1.size());
    multNxNTransposed(m1, m2, mask, Cs);
    return Cs;
}

///////
// #normalize
void normalizeInplace(MemoryView<Quaternion, VRAM_CUDA>& q)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (q.size() + blockSize - 1) / blockSize;
    cuda::normalizeInplace_kernel<<<gridSize, blockSize>>>(q.raw(), q.size());
    RM_CUDA_DEBUG();
}

///////
// #setter

namespace cuda
{

template<typename T>
__global__ void setIdentity_kernel(
    T* data,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        data[id] = T::Identity();
    }
}

} // namespace cuda

void setIdentity(MemoryView<Quaternion, VRAM_CUDA>& qs)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (qs.size() + blockSize - 1) / blockSize;
    cuda::setIdentity_kernel<<<gridSize, blockSize>>>(qs.raw(), qs.size());
    RM_CUDA_DEBUG();
}

void setIdentity(MemoryView<Transform, VRAM_CUDA>& Ts)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (Ts.size() + blockSize - 1) / blockSize;
    cuda::setIdentity_kernel<<<gridSize, blockSize>>>(Ts.raw(), Ts.size());
    RM_CUDA_DEBUG();
}

void setIdentity(MemoryView<Matrix3x3, VRAM_CUDA>& Ms)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (Ms.size() + blockSize - 1) / blockSize;
    cuda::setIdentity_kernel<<<gridSize, blockSize>>>(Ms.raw(), Ms.size());
    RM_CUDA_DEBUG();
}

void setIdentity(MemoryView<Matrix4x4, VRAM_CUDA>& Ms)
{
    constexpr unsigned int blockSize = 1024;
    const unsigned int gridSize = (Ms.size() + blockSize - 1) / blockSize;
    cuda::setIdentity_kernel<<<gridSize, blockSize>>>(Ms.raw(), Ms.size());
    RM_CUDA_DEBUG();
}

void setZeros(MemoryView<Matrix3x3, VRAM_CUDA>& Ms)
{
    cudaMemset(Ms.raw(), 0, Ms.size() * sizeof(Matrix3x3) );
}

void setZeros(MemoryView<Matrix4x4, VRAM_CUDA>& Ms)
{
    cudaMemset(Ms.raw(), 0, Ms.size() * sizeof(Matrix4x4) );
}

namespace cuda
{
//////////
// #sum
// TODO: check perfomance of sum_kernel
template<unsigned int nMemElems, typename T>
__global__ void sum_kernel(
    const T* data,
    T* res,
    unsigned int N)
{
    // sharedMemElements per block

    // Many blocks stategy
    // rows=2
    // 
    //   blockId=0                  blockId=1
    // sharedMemElements |  -- sharedMemElements --- 
    // [ 1,  3,  5,  7]    [9,  11, 13, 15]
    // [ 2,  4,  6,  8]    [10, 12, 14, 16]
    //   |   |   |   |       |   |   |   | 
    // [ 3,  7, 11, 15]    [19, 23, 27, 31]
    // [10, 26]            [42, 58]
    // [36]                [100]
    __shared__ T sdata[nMemElems];

    const unsigned int n_threads = blockDim.x;
    const unsigned int n_blocks = gridDim.x;

    const unsigned int total_threads = n_threads * n_blocks;
    const unsigned int n_rows = (N + total_threads - 1) / total_threads;
    const unsigned int n_elems_per_block = n_rows * nMemElems;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int glob_shift = n_elems_per_block * bid;

    sdata[tid] *= 0.0;
    for(unsigned int i=0; i<n_rows; i++)
    {
        const unsigned int data_id = glob_shift + i * nMemElems + tid; // advance one row
        if(data_id < N)
        {
            sdata[tid] += data[data_id];
        }
    }
    __syncthreads();

    for(unsigned int s = nMemElems / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        } else {
            // TODO: can this thread do something useful in the meantime?
        }
        __syncthreads();
    }

    if(tid < nMemElems / 2 && tid < 32)
    {
        warpReduce<nMemElems>(sdata, tid);
    }

    // Do this instead for types that have no volotile operators implemented:
    // for(unsigned int s = nMemElems / 2; s > 0; s >>= 1)
    // {
    //     if(tid < s)
    //     {
    //         sdata[tid] += sdata[tid + s];
    //     } else {
    //         // TODO: can this thread do something useful in the meantime?
    //     }
    //     __syncthreads();
    // }
    // the warpReduce gives a comparable small performance boost with my profiling tests
    // so, this would be still a good we to do a reduction

    if(tid == 0)
    {
        res[bid] = sdata[0];
    }
}

} // namespace cuda

void sum(
    const MemoryView<Vector, VRAM_CUDA>& data,
    MemoryView<Vector, VRAM_CUDA>& s)
{
    const unsigned int n_outputs = s.size(); // also number of blocks
    constexpr unsigned int n_threads = 1024; // also shared mem
    // the rest is computed automatically

    cuda::sum_kernel<n_threads> <<<n_outputs, n_threads>>>(data.raw(), s.raw(), data.size());
    RM_CUDA_DEBUG();
}

Memory<Vector, VRAM_CUDA> sum(
    const MemoryView<Vector, VRAM_CUDA>& data)
{
    Memory<Vector, VRAM_CUDA> s(1);
    sum(data, s);
    return s;
}

void sum(
    const MemoryView<int, VRAM_CUDA>& data,
    MemoryView<int, VRAM_CUDA>& s)
{
    const unsigned int n_outputs = s.size(); // also number of blocks
    constexpr unsigned int n_threads = 1024; // also shared mem

    cuda::sum_kernel<n_threads> <<<n_outputs, n_threads>>>(data.raw(), s.raw(), data.size());
    RM_CUDA_DEBUG();
}

Memory<int, VRAM_CUDA> sum(
    const MemoryView<int, VRAM_CUDA>& data)
{
    Memory<int, VRAM_CUDA> s(1);
    sum(data, s);
    return s;
}

//////////
// #mean
void mean(
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& res)
{
    sum(X, res);
    divNx1Inplace(res, X.size());
}

Memory<Vector, VRAM_CUDA> mean(
    const MemoryView<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> res(1);
    mean(X, res);
    return res;
}

//////////
// #cov
void cov(
    const MemoryView<Vector, VRAM_CUDA>& v1,
    const MemoryView<Vector, VRAM_CUDA>& v2,
    MemoryView<Matrix3x3, VRAM_CUDA>& C)
{
    cuda::cov_kernel<1024> <<<1, 1024>>>(v1.raw(), v2.raw(), C.raw(), v1.size());
    RM_CUDA_DEBUG();
}

Memory<Matrix3x3, VRAM_CUDA> cov(
    const MemoryView<Vector, VRAM_CUDA>& v1,
    const MemoryView<Vector, VRAM_CUDA>& v2
)
{
    Memory<Matrix3x3, VRAM_CUDA> C(1);
    cov(v1, v2, C);
    return C;
}

namespace cuda
{

__global__ void svd_kernel(
    const Matrix3x3* As,
    Matrix3x3* Us,
    Matrix3x3* Ws,
    Matrix3x3* Vs, 
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        svd(As[id], Us[id], Ws[id], Vs[id]);
    }
}

} // namespace cuda

void svd(
    const MemoryView<Matrix3x3, VRAM_CUDA>& As,
    MemoryView<Matrix3x3, VRAM_CUDA>& Us,
    MemoryView<Matrix3x3, VRAM_CUDA>& Ws,
    MemoryView<Matrix3x3, VRAM_CUDA>& Vs
)
{
    constexpr unsigned int blockSize = 512;
    const unsigned int gridSize = (As.size() + blockSize - 1) / blockSize;
    cuda::svd_kernel<<<gridSize, blockSize>>>(As.raw(), Us.raw(), Ws.raw(), Vs.raw(), As.size());
    RM_CUDA_DEBUG();
}

namespace cuda
{

__global__ void svd_kernel(
    const Matrix3x3* As,
    Matrix3x3* Us,
    Vector3* ws,
    Matrix3x3* Vs,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        svd(As[id], Us[id], ws[id], Vs[id]);
    }
}

} // namespace cuda

void svd(
    const MemoryView<Matrix3x3, VRAM_CUDA>& As,
    MemoryView<Matrix3x3, VRAM_CUDA>& Us,
    MemoryView<Vector3, VRAM_CUDA>& ws,
    MemoryView<Matrix3x3, VRAM_CUDA>& Vs
)
{
    constexpr unsigned int blockSize = 512;
    const unsigned int gridSize = (As.size() + blockSize - 1) / blockSize;
    cuda::svd_kernel<<<gridSize, blockSize>>>(As.raw(), Us.raw(), ws.raw(), Vs.raw(), As.size());
    RM_CUDA_DEBUG();
}

namespace cuda
{
__global__ void umeyama_transform_kernel(
    Transform* Ts,
    const Vector3* ds,
    const Vector3* ms,
    const Matrix3x3* Cs,
    const unsigned int* n_meas,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        Ts[id] = umeyama_transform(ds[id], ms[id], Cs[id], n_meas[id]);
    }
}
} // namespace cuda

void umeyama_transform(
    MemoryView<Transform, VRAM_CUDA>& Ts,
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs,
    const MemoryView<unsigned int, VRAM_CUDA>& n_meas)
{
    constexpr unsigned int blockSize = 256;
    const unsigned int gridSize = (Ts.size() + blockSize - 1) / blockSize;
    cuda::umeyama_transform_kernel<<<gridSize, blockSize>>>(Ts.raw(), ds.raw(), ms.raw(), Cs.raw(), n_meas.raw(), Ts.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> umeyama_transform(
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs,
    const MemoryView<unsigned int, VRAM_CUDA>& n_meas)
{
    Memory<Transform, VRAM_CUDA> ret(ds.size());
    umeyama_transform(ret, ds, ms, Cs, n_meas);
    return ret;
}

namespace cuda
{
__global__ void umeyama_transform_kernel(
    Transform* Ts,
    const Vector3* ds,
    const Vector3* ms,
    const Matrix3x3* Cs,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        Ts[id] = umeyama_transform(ds[id], ms[id], Cs[id]);
    }
}
} // namespace cuda

void umeyama_transform(
    MemoryView<Transform, VRAM_CUDA>& Ts,
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs)
{
    constexpr unsigned int blockSize = 256;
    const unsigned int gridSize = (Ts.size() + blockSize - 1) / blockSize;
    cuda::umeyama_transform_kernel<<<gridSize, blockSize>>>(Ts.raw(), ds.raw(), ms.raw(), Cs.raw(), Ts.size());
    RM_CUDA_DEBUG();
}

Memory<Transform, VRAM_CUDA> umeyama_transform(
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs)
{
    Memory<Transform, VRAM_CUDA> ret(ds.size());
    umeyama_transform(ret, ds, ms, Cs);
    return ret;
}

namespace cuda
{

template<unsigned int nMemElems>
__global__ void print_variables(
    const int* data,
    int* res,
    unsigned int N)
{
    const unsigned int n_threads_per_block = blockDim.x;
    const unsigned int n_blocks = gridDim.x;

    const unsigned int n_threads_total = n_threads_per_block * n_blocks;
    const unsigned int n_rows_per_block = (N + n_threads_total - 1) / n_threads_total;
    const unsigned int n_elems_per_block = n_rows_per_block * nMemElems;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int block_shift = n_elems_per_block * bid;

    if(tid == 0 && bid == 0)
    {
        printf("Variables:\n");
        printf("- # blocks, threads: %u, %u\n", n_blocks, n_threads_per_block);
        printf("- # rows: %u\n", n_rows_per_block);
        printf("- # elems per block: %u\n", n_elems_per_block);
        printf("- block shift: %u\n", block_shift);
    }
}

//////////
// #sum
// TODO: check perfomance of sum_kernel
template<unsigned int nMemElems>
__global__ void sum_kernel_test(
    const int* data,
    int* res,
    unsigned int N)
{
    // sharedMemElements per block

    // Many blocks stategy
    // rows=2
    // 
    //   blockId=0                  blockId=1
    // sharedMemElements |  -- sharedMemElements --- 
    // [ 1,  3,  5,  7]    [9,  11, 13, 15]
    // [ 2,  4,  6,  8]    [10, 12, 14, 16]
    //   |   |   |   |       |   |   |   | 
    // [ 3,  7, 11, 15]    [19, 23, 27, 31]
    // [10, 26]            [42, 58]
    // [36]                [100]
    __shared__ int sdata[nMemElems];

    const unsigned int n_threads_per_block = blockDim.x;
    const unsigned int n_blocks = gridDim.x;

    const unsigned int n_threads_total = n_threads_per_block * n_blocks;
    const unsigned int n_rows_per_block = (N + n_threads_total - 1) / n_threads_total;
    const unsigned int n_elems_per_block = n_rows_per_block * nMemElems;

    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int block_shift = n_elems_per_block * bid;

    sdata[tid] = 0;
    for(unsigned int i=0; i<n_rows_per_block; i++)
    {
        const unsigned int data_id = block_shift + i * nMemElems + tid; // advance one row
        if(data_id < N)
        {
            printf("b%u, t%u: data -> smem: %u -> %u\n", bid, tid, data_id, tid);
            sdata[tid] += data[data_id];
        }
    }
    __syncthreads();

    unsigned int depth = 0;
    for(unsigned int s = nMemElems / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            printf("b%u, t%u: smem reduce: (%u + %u)_%u -> (%u)_%u\n", bid, tid, tid, tid + s, depth, tid, depth+1);
            sdata[tid] += sdata[tid + s];
        } else {
            // TODO: can this thread do something useful in the meantime?
        }
        depth++;
        __syncthreads();
    }

    if(tid == 0)
    {
        res[bid] = sdata[0];
    }
}

} // namespace cuda

void sum_reduce_test_t1(
    const MemoryView<int, VRAM_CUDA>& data, 
    MemoryView<int, VRAM_CUDA> results)
{
    const unsigned int n_outputs = results.size(); // also number of blocks
    constexpr unsigned int n_threads = 1; // also shared mem
    cuda::print_variables<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
    cuda::sum_kernel_test<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
}

void sum_reduce_test_t2(
    const MemoryView<int, VRAM_CUDA>& data, 
    MemoryView<int, VRAM_CUDA> results)
{
    const unsigned int n_outputs = results.size(); // also number of blocks
    constexpr unsigned int n_threads = 2; // also shared mem
    cuda::print_variables<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
    cuda::sum_kernel_test<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
}

void sum_reduce_test_t4(
    const MemoryView<int, VRAM_CUDA>& data, 
    MemoryView<int, VRAM_CUDA> results)
{
    const unsigned int n_outputs = results.size(); // also number of blocks
    constexpr unsigned int n_threads = 4; // also shared mem
    cuda::print_variables<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
    cuda::sum_kernel_test<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
}

void sum_reduce_test_t8(
    const MemoryView<int, VRAM_CUDA>& data, 
    MemoryView<int, VRAM_CUDA> results)
{
    const unsigned int n_outputs = results.size(); // also number of blocks
    constexpr unsigned int n_threads = 8; // also shared mem
    cuda::print_variables<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
    cuda::sum_kernel_test<n_threads> <<<n_outputs, n_threads>>>(data.raw(), results.raw(), data.size());
}

} // namespace rmagine